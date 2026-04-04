#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain Tumor Detection Inference Script
=======================================
Uses the preprocessing pipeline from Preprocessing_for_prediction.py.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm

# Import existing preprocessing class
from Preprocessing_for_prediction import MultimodalBrainTumorPreprocessingPipeline

# ============================================================================
# CUSTOM LAYER REGISTRATION
# ============================================================================
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class CorrelationLayer(tf.keras.layers.Layer):
    """Custom layer to compute correlation matrix of patch features."""
    def call(self, inputs):
        return tf.matmul(inputs, inputs, transpose_b=True)


@register_keras_serializable()
class PatchExtractorLayer(tf.keras.layers.Layer):
    """Custom layer to extract patches using tf.image.extract_patches."""
    def __init__(self, patch_size=28, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):
        return tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size})
        return config

# ============================================================================
# CONFIGURATION (adjust paths if needed)
# ============================================================================
MODEL_PATHS = {
    'mri_stage1': './Saved_models/MRI_stage1_binary_model.keras',
    'ct_correlation': './Saved_models/ct_correlation_model.keras',
    'fusion': './Saved_models/fusion_model.keras'
}

# Input shapes expected by each model
MRI_MODEL_SHAPE = (128, 128, 4)
CT_MODEL_SHAPE   = (224, 224, 4)

CLASS_NAMES = ['Normal', 'Tumor']

# ============================================================================
# Grad‑CAM helper – recursive last conv layer finder
# ============================================================================
def find_last_conv_layer(model):
    """Recursively find the last Conv2D layer in the model (including submodels)."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        if hasattr(layer, 'layers'):
            sub_last = find_last_conv_layer(layer)
            if sub_last:
                last_conv = sub_last
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")
    return last_conv

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
def load_mri_model():
    """Load the pre‑trained MRI stage‑1 model (binary classifier)."""
    if not os.path.exists(MODEL_PATHS['mri_stage1']):
        raise FileNotFoundError(f"MRI model not found at {MODEL_PATHS['mri_stage1']}")
    model = tf.keras.models.load_model(MODEL_PATHS['mri_stage1'], safe_mode=False)
    print("✓ MRI model loaded.")
    return model

def build_ct_correlation_model(input_shape, patch_size=28, num_patches=64, num_classes=2):
    """Reconstruct the CT correlation model architecture (required for loading weights)."""
    inputs = layers.Input(shape=input_shape, name="ct_input")

    # Use the custom patch extractor layer
    extract_patches = PatchExtractorLayer(patch_size=patch_size, name="patch_extractor")
    patches = extract_patches(inputs)

    patches_reshaped = layers.Reshape((num_patches, -1), name="reshape_flat")(patches)
    patches_img = layers.Reshape((num_patches, patch_size, patch_size, input_shape[-1]),
                                  name="reshape_image")(patches_reshaped)

    encoder = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(patch_size, patch_size, input_shape[-1]),
                      name="enc_conv1"),
        layers.MaxPooling2D((2, 2), name="enc_pool1"),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="enc_conv2"),
        layers.MaxPooling2D((2, 2), name="enc_pool2"),
        layers.Flatten(name="enc_flatten"),
        layers.Dense(128, activation='relu', name="enc_dense"),
        layers.LayerNormalization(name="enc_layernorm")
    ], name="shared_encoder")

    patch_features = layers.TimeDistributed(encoder, name="time_distributed_encoder")(patches_img)
    correlation_matrix = CorrelationLayer(name="correlation_layer")(patch_features)
    guided_features = layers.Dot(axes=(2, 1), name="guided_features")([correlation_matrix, patch_features])
    combined_features = layers.Concatenate(axis=-1, name="combined_features")([patch_features, guided_features])
    pooled_features = layers.GlobalAveragePooling1D(name="global_pool")(combined_features)
    x = layers.Dense(256, activation='relu', name="fc1")(pooled_features)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(32, activation='relu', name="fc2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CT_Correlation_Model")
    return model

def load_ct_model():
    """Load the pre‑trained CT correlation model (architecture + weights)."""
    if not os.path.exists(MODEL_PATHS['ct_correlation']):
        raise FileNotFoundError(f"CT model not found at {MODEL_PATHS['ct_correlation']}")
    model = build_ct_correlation_model(CT_MODEL_SHAPE, patch_size=28, num_patches=64, num_classes=2)
    model.load_weights(MODEL_PATHS['ct_correlation'])
    print("✓ CT model loaded.")
    return model

def load_fusion_model():
    """Load the pre‑trained multimodal fusion model with custom objects."""
    if not os.path.exists(MODEL_PATHS['fusion']):
        raise FileNotFoundError(f"Fusion model not found at {MODEL_PATHS['fusion']}")
    # Provide all custom classes used in the saved model
    custom_objects = {
        'CorrelationLayer': CorrelationLayer,
        'PatchExtractorLayer': PatchExtractorLayer,
    }
    model = tf.keras.models.load_model(
        MODEL_PATHS['fusion'],
        custom_objects=custom_objects,
        safe_mode=False
    )
    print("✓ Fusion model loaded.")
    return model

# ============================================================================
# INFERENCE HELPERS
# ============================================================================
def predict_mri(model, image_batch):
    probs = model.predict(image_batch, verbose=0)[0]
    return np.argmax(probs), probs[1]

def predict_ct(model, image_batch):
    probs = model.predict(image_batch, verbose=0)[0]
    return np.argmax(probs), probs[1]

def predict_fusion(model, mri_batch, ct_batch):
    probs = model.predict([mri_batch, ct_batch], verbose=0)[0]
    return np.argmax(probs), probs[1]

def preprocess_single_image(image_path, modality, pipeline):
    """
    Use the pipeline to preprocess a single image.
    The pipeline returns a normalized (0‑1) multi‑scale image of shape (224,224,4).
    For MRI we resize to (128,128,4) to match the model input.
    """
    proc = pipeline.modality_specific_preprocessing(
        image_path,
        modality=modality
    )
    if proc is None:
        return None

    if modality == "MRI":
        # Resize spatial dimensions to MRI model input (128x128)
        proc_resized = cv2.resize(proc, MRI_MODEL_SHAPE[:2])
        # Ensure shape is exactly (128,128,4)
        proc_resized = proc_resized.reshape(*MRI_MODEL_SHAPE)
    else:  # CT – shape is already (224,224,4)
        proc_resized = proc

    # Add batch dimension
    return np.expand_dims(proc_resized, axis=0).astype(np.float32)

# ============================================================================
# MAIN INTERACTIVE FUNCTION (optional CLI)
# ============================================================================
def main():
    print("=" * 60)
    print("        BRAIN TUMOR DETECTION INFERENCE")
    print("           (with on‑the‑fly preprocessing)")
    print("=" * 60)

    # 1. Ask user which modality they have
    print("\nWhich modality do you have?")
    print("  1) MRI only")
    print("  2) CT only")
    print("  3) Both (MRI and CT)")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice not in ['1', '2', '3']:
        print("Invalid choice. Exiting.")
        return

    # 2. Instantiate preprocessing pipeline 
    pipeline = MultimodalBrainTumorPreprocessingPipeline(target_size=(224, 224))

    # 3. Load required models
    try:
        if choice == '1':
            model_mri = load_mri_model()
        elif choice == '2':
            model_ct = load_ct_model()
        else:  # both
            model_mri = load_mri_model()
            model_ct = load_ct_model()
            model_fusion = load_fusion_model()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. Supported image extensions
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    results = []

    if choice == '1':
        # ---------- MRI only ----------
        folder = input("Enter path to folder containing MRI images: ").strip()
        if not os.path.isdir(folder):
            print("Folder does not exist.")
            return
        files = [f for f in glob(os.path.join(folder, "*")) if f.lower().endswith(valid_ext)]
        if not files:
            print("No supported image files found.")
            return
        print(f"Found {len(files)} MRI files.")

        for f in tqdm(files, desc="Processing MRI"):
            batch = preprocess_single_image(f, "MRI", pipeline)
            if batch is None:
                print(f"Warning: could not preprocess {f}, skipping.")
                continue
            pred_class, tumor_prob = predict_mri(model_mri, batch)
            results.append((os.path.basename(f), pred_class, tumor_prob))

    elif choice == '2':
        # ---------- CT only ----------
        folder = input("Enter path to folder containing CT images: ").strip()
        if not os.path.isdir(folder):
            print("Folder does not exist.")
            return
        files = [f for f in glob(os.path.join(folder, "*")) if f.lower().endswith(valid_ext)]
        if not files:
            print("No supported image files found.")
            return
        print(f"Found {len(files)} CT files.")

        for f in tqdm(files, desc="Processing CT"):
            batch = preprocess_single_image(f, "CT", pipeline)
            if batch is None:
                print(f"Warning: could not preprocess {f}, skipping.")
                continue
            pred_class, tumor_prob = predict_ct(model_ct, batch)
            results.append((os.path.basename(f), pred_class, tumor_prob))
            
    else:
        # ---------- Both (MRI and CT) ----------
        mri_folder = input("Enter path to folder containing MRI images: ").strip()
        ct_folder  = input("Enter path to folder containing CT images: ").strip()
        if not os.path.isdir(mri_folder) or not os.path.isdir(ct_folder):
            print("One or both folders do not exist.")
            return

        mri_files = [f for f in glob(os.path.join(mri_folder, "*")) if f.lower().endswith(valid_ext)]
        ct_files  = [f for f in glob(os.path.join(ct_folder,  "*")) if f.lower().endswith(valid_ext)]
        
        if not mri_files or not ct_files:
            print("Missing image files in one or both folders.")
            return

        def basename_no_ext(path):
            return os.path.splitext(os.path.basename(path))[0]

        mri_bases = {basename_no_ext(f): f for f in mri_files}
        ct_bases  = {basename_no_ext(f): f for f in ct_files}

        common_bases = sorted(set(mri_bases.keys()) & set(ct_bases.keys()))

        if common_bases:
            print(f"Found {len(common_bases)} paired images. Using FUSION model.")
            for base in tqdm(common_bases, desc="Processing pairs"):
                mri_batch = preprocess_single_image(mri_bases[base], "MRI", pipeline)
                ct_batch  = preprocess_single_image(ct_bases[base], "CT", pipeline)
                
                if mri_batch is None or ct_batch is None:
                    print(f"Warning: preprocessing failed for {base}, skipping.")
                    continue

                pred_class, tumor_prob = predict_fusion(model_fusion, mri_batch, ct_batch)
                results.append((base, pred_class, tumor_prob))
        else:
            print("No common filenames found. Fusion not possible. Processing separately...")
            results_mri, results_ct = [], []
            
            for f in tqdm(mri_files, desc="Processing MRI"):
                batch = preprocess_single_image(f, "MRI", pipeline)
                if batch is not None:
                    pred_class, tumor_prob = predict_mri(model_mri, batch)
                    results_mri.append((os.path.basename(f), pred_class, tumor_prob))

            for f in tqdm(ct_files, desc="Processing CT"):
                batch = preprocess_single_image(f, "CT", pipeline)
                if batch is not None:
                    pred_class, tumor_prob = predict_ct(model_ct, batch)
                    results_ct.append((os.path.basename(f), pred_class, tumor_prob))

            print("\n--- MRI Predictions ---")
            for name, cls, prob in results_mri:
                print(f"{name:30s} : {CLASS_NAMES[cls]:8s} (tumor prob: {prob:.4f})")
            print("\n--- CT Predictions ---")
            for name, cls, prob in results_ct:
                print(f"{name:30s} : {CLASS_NAMES[cls]:8s} (tumor prob: {prob:.4f})")
            return

    # 5. Print Output Results (For Single Modality or Paired Fusions)
    if results:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        tumor_count = sum(1 for _, cls, _ in results if cls == 1)
        
        for name, cls, prob in results:
            print(f"{name:30s} : {CLASS_NAMES[cls]:8s} (tumor prob: {prob:.4f})")

        print("\n" + "-" * 40)
        print(f"Total images processed: {len(results)}")
        print(f"Tumor detected: {tumor_count}")
        print(f"Normal detected: {len(results) - tumor_count}")
        print(f"Average tumor probability: {np.mean([prob for _, _, prob in results]):.4f}")

    print("\nInference complete.")

if __name__ == "__main__":
    main()