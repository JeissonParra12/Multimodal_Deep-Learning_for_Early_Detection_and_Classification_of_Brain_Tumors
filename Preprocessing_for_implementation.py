#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain Tumor Detection Inference Script
=======================================
Las capas custom estan definidas aqui directamente con el mismo
package="CTClassifier" que se uso en CT_Classifier al entrenar.
Keras las empareja por (package, class_name), no por donde estan
definidas, asi que load_model funciona sin ningun import externo.
"""
 
import os
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm
from tensorflow.keras.saving import register_keras_serializable
 
from Preprocessing_for_prediction import MultimodalBrainTumorPreprocessingPipeline
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Custom layers — mismo package="CTClassifier" que CT_Classifier.ipynb
# Keras los empareja por nombre al hacer load_model()
# ─────────────────────────────────────────────────────────────────────────────
 
@register_keras_serializable(package="CTClassifier")
class PatchExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size=28, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
 
    def call(self, inputs):
        return tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
 
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
 
 
@register_keras_serializable(package="CTClassifier")
class PatchStatisticsLayer(tf.keras.layers.Layer):
    def call(self, patch_features):
        p_min  = tf.reduce_min (patch_features, axis=-1, keepdims=True)
        p_max  = tf.reduce_max (patch_features, axis=-1, keepdims=True)
        p_sum  = tf.reduce_sum (patch_features, axis=-1, keepdims=True)
        p_mean = tf.reduce_mean(patch_features, axis=-1, keepdims=True)
        p_std  = tf.math.reduce_std(patch_features, axis=-1, keepdims=True)
        sorted_f = tf.sort(patch_features, axis=-1)
        n        = tf.shape(patch_features)[-1]
        p_median = tf.reduce_mean(
            sorted_f[:, :, n // 4 : 3 * n // 4], axis=-1, keepdims=True)
        return tf.concat(
            [p_min, p_max, p_sum, p_mean, p_std, p_median], axis=-1)
 
    def get_config(self):
        return super().get_config()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    'mri_stage1':     './Saved_models/MRI_stage1_binary_model.keras',
    'ct_correlation': './Saved_models/ct_correlation_model.keras',
    'fusion':         './Saved_models/fusion_model.keras',
}
 
MRI_MODEL_SHAPE = (224, 224, 4)
CT_MODEL_SHAPE  = (224, 224, 4)
CLASS_NAMES     = ['Normal', 'Tumor']
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Model loaders — load_model puro, sin reconstruir arquitectura
# ─────────────────────────────────────────────────────────────────────────────
def load_mri_model():
    if not os.path.exists(MODEL_PATHS['mri_stage1']):
        raise FileNotFoundError(MODEL_PATHS['mri_stage1'])
    return tf.keras.models.load_model(
        MODEL_PATHS['mri_stage1'], compile=False, safe_mode=False)
 
 
def load_ct_model():
    if not os.path.exists(MODEL_PATHS['ct_correlation']):
        raise FileNotFoundError(MODEL_PATHS['ct_correlation'])
    return tf.keras.models.load_model(
        MODEL_PATHS['ct_correlation'], compile=False, safe_mode=False)
 
 
def load_fusion_model():
    if not os.path.exists(MODEL_PATHS['fusion']):
        raise FileNotFoundError(MODEL_PATHS['fusion'])
    return tf.keras.models.load_model(
        MODEL_PATHS['fusion'], compile=False, safe_mode=False)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM helper
# ─────────────────────────────────────────────────────────────────────────────
def find_last_conv_layer(model):
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        if hasattr(layer, 'layers'):
            sub = find_last_conv_layer(layer)
            if sub:
                last_conv = sub
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")
    return last_conv
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_mri(model, image_batch):
    probs = model.predict(image_batch, verbose=0)[0]
    return int(np.argmax(probs)), float(probs[1])
 
 
def predict_ct(model, image_batch):
    probs = model.predict(image_batch, verbose=0)[0]
    return int(np.argmax(probs)), float(probs[1])
 
 
def predict_fusion(model, mri_batch, ct_batch):
    probs = model.predict([mri_batch, ct_batch], verbose=0)[0]
    return int(np.argmax(probs)), float(probs[1])
 
 
def preprocess_single_image(image_path, modality, pipeline):
    proc = pipeline.modality_specific_preprocessing(
        image_path, modality=modality)
    if proc is None:
        return None
    if modality == "MRI":
        proc = cv2.resize(proc, MRI_MODEL_SHAPE[:2])
        proc = proc.reshape(*MRI_MODEL_SHAPE)
    return np.expand_dims(proc, axis=0).astype(np.float32)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("        BRAIN TUMOR DETECTION INFERENCE")
    print("=" * 60)
 
    print("\nWhich modality do you have?")
    print("  1) MRI only")
    print("  2) CT only")
    print("  3) Both (MRI and CT)")
    choice = input("Enter 1, 2, or 3: ").strip()
 
    if choice not in ("1", "2", "3"):
        print("Invalid choice. Exiting.")
        return
 
    pipeline = MultimodalBrainTumorPreprocessingPipeline(target_size=(224, 224))
 
    try:
        if choice == "1":
            model_mri = load_mri_model()
        elif choice == "2":
            model_ct = load_ct_model()
        else:
            model_mri    = load_mri_model()
            model_ct     = load_ct_model()
            model_fusion = load_fusion_model()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
 
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    results   = []
 
    if choice == "1":
        folder = input("Path to folder with MRI images: ").strip()
        if not os.path.isdir(folder):
            print("Folder not found."); return
        files = [f for f in glob(os.path.join(folder, "*"))
                 if f.lower().endswith(valid_ext)]
        print(f"Found {len(files)} MRI files.")
        for f in tqdm(files, desc="Processing MRI"):
            b = preprocess_single_image(f, "MRI", pipeline)
            if b is None:
                print(f"  Skipping: {f}"); continue
            cls, prob = predict_mri(model_mri, b)
            results.append((os.path.basename(f), cls, prob))
 
    elif choice == "2":
        folder = input("Path to folder with CT images: ").strip()
        if not os.path.isdir(folder):
            print("Folder not found."); return
        files = [f for f in glob(os.path.join(folder, "*"))
                 if f.lower().endswith(valid_ext)]
        print(f"Found {len(files)} CT files.")
        for f in tqdm(files, desc="Processing CT"):
            b = preprocess_single_image(f, "CT", pipeline)
            if b is None:
                print(f"  Skipping: {f}"); continue
            cls, prob = predict_ct(model_ct, b)
            results.append((os.path.basename(f), cls, prob))
 
    else:
        mri_folder = input("Path to folder with MRI images: ").strip()
        ct_folder  = input("Path to folder with CT images:  ").strip()
        if not os.path.isdir(mri_folder) or not os.path.isdir(ct_folder):
            print("One or both folders not found."); return
 
        def bname(p):
            return os.path.splitext(os.path.basename(p))[0]
 
        mri_map = {bname(f): f for f in glob(os.path.join(mri_folder, "*"))
                   if f.lower().endswith(valid_ext)}
        ct_map  = {bname(f): f for f in glob(os.path.join(ct_folder, "*"))
                   if f.lower().endswith(valid_ext)}
        common  = sorted(mri_map.keys() & ct_map.keys())
 
        if common:
            print(f"Found {len(common)} paired images => Fusion model.")
            for base in tqdm(common, desc="Fusion"):
                mb = preprocess_single_image(mri_map[base], "MRI", pipeline)
                cb = preprocess_single_image(ct_map[base],  "CT",  pipeline)
                if mb is None or cb is None:
                    print(f"  Skipping: {base}"); continue
                cls, prob = predict_fusion(model_fusion, mb, cb)
                results.append((base, cls, prob))
        else:
            print("No common filenames - processing separately.")
            for f in tqdm(list(mri_map.values()), desc="MRI"):
                b = preprocess_single_image(f, "MRI", pipeline)
                if b is not None:
                    cls, prob = predict_mri(model_mri, b)
                    results.append((os.path.basename(f), cls, prob))
            for f in tqdm(list(ct_map.values()), desc="CT"):
                b = preprocess_single_image(f, "CT", pipeline)
                if b is not None:
                    cls, prob = predict_ct(model_ct, b)
                    results.append((os.path.basename(f), cls, prob))
 
    if results:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        tumors = sum(1 for _, c, _ in results if c == 1)
        for name, cls, prob in results:
            print(f"{name:40s}: {CLASS_NAMES[cls]:8s}  (tumor prob: {prob:.4f})")
        print("\n" + "-" * 40)
        print(f"Total processed : {len(results)}")
        print(f"Tumor detected  : {tumors}")
        print(f"Normal detected : {len(results) - tumors}")
        print(f"Avg tumor prob  : {np.mean([p for _,_,p in results]):.4f}")
 
    print("\nInference complete.")
 
 
if __name__ == "__main__":
    main()