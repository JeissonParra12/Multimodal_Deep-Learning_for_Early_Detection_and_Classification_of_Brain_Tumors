"""
Streamlit Web App for Brain Tumor Detection with Red Circle Localization
========================================================================
If a tumor is detected, a red circle is drawn on the preprocessed image
(first channel) highlighting the suspicious region.
Supports single‑image upload, multi‑file batch processing, and precomputed
performance graphs from saved images.
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import tensorflow as tf
from pathlib import Path
from glob import glob

# Import your existing modules
from Preprocessing_for_prediction import MultimodalBrainTumorPreprocessingPipeline
from inference import (
    load_mri_model,
    load_ct_model,
    load_fusion_model,
    predict_mri,
    predict_ct,
    predict_fusion,
    preprocess_single_image,
    MRI_MODEL_SHAPE,
    CT_MODEL_SHAPE,
    CLASS_NAMES,
    CorrelationLayer,       # still needed for custom objects
    MODEL_PATHS,
)

# Define additional model paths (add Stage 2)
STAGE2_MODEL_PATH = './Saved_models/MRI_stage2_multiclass_model.keras'
STAGE2_CLASS_NAMES = ['Meningioma', 'Glioma', 'Pituitary']

# ----------------------------------------------------------------------
# Occlusion‑based Localization (robust for any model)
# ----------------------------------------------------------------------
def generate_occlusion_heatmap(model, image_batch, class_index, patch_size=32, stride=16):
    """
    Architecture-agnostic tumor localization.
    Slides a mask over the image and measures the drop in tumor probability.
    """
    # Ensure we are working with a numpy array
    img = image_batch[0].numpy() if isinstance(image_batch, tf.Tensor) else image_batch[0]
    h, w, c = img.shape
    
    # 1. Get baseline prediction
    baseline_preds = model.predict(image_batch, verbose=0)[0]
    baseline_prob = baseline_preds[class_index]
    
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    # 2. Generate all masked variations of the image
    masked_images = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            masked_img = img.copy()
            # Zero out the sliding patch (hiding that part of the brain)
            masked_img[y:y+patch_size, x:x+patch_size, :] = 0 
            masked_images.append(masked_img)
            coords.append((y, x))
            
    # 3. Predict all masked images in one fast batch
    masked_batch = np.array(masked_images)
    preds = model.predict(masked_batch, batch_size=32, verbose=0)
    
    # 4. Calculate probability drop
    for i, (y, x) in enumerate(coords):
        # How much did hiding this region hurt the model's confidence?
        prob_drop = baseline_prob - preds[i, class_index]
        
        # We only care about regions that *lowered* the confidence when hidden
        prob_drop = max(0, prob_drop) 
        
        heatmap[y:y+patch_size, x:x+patch_size] += prob_drop
        counts[y:y+patch_size, x:x+patch_size] += 1
        
    # 5. Average overlapping regions and normalize
    counts[counts == 0] = 1 # avoid division by zero
    heatmap = heatmap / counts
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    return heatmap


def draw_tumor_circle(image, heatmap, min_radius=10, max_radius=50):
    """Draw a red circle on the image at the region of highest activation."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    _, maxVal, _, maxLoc = cv2.minMaxLoc(heatmap_resized)
    center = maxLoc

    threshold = 0.8 * maxVal
    mask = (heatmap_resized >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(largest)
        radius = int(radius)
        radius = max(min_radius, min(radius, max_radius))
    else:
        diagonal = np.sqrt(h**2 + w**2)
        radius = int(diagonal * 0.1)
        radius = max(min_radius, min(radius, max_radius))

    overlay = image.copy()
    cv2.circle(overlay, center, radius, (0, 0, 255), thickness=3)
    return overlay


def get_display_image(processed_batch, modality):
    """Extract first channel of preprocessed 4‑channel array for display."""
    img = processed_batch[0]          # (H,W,4)
    if modality == "MRI":
        img = cv2.resize(img, (224, 224))
    display = (img[:, :, 0] * 255).astype(np.uint8)
    return display


# ----------------------------------------------------------------------
# Model Loading with Caching
# ----------------------------------------------------------------------
@st.cache_resource
def load_models(modality):
    if modality == "MRI only":
        mri = load_mri_model()
        ct = None
        fusion = None
    elif modality == "CT only":
        mri = None
        ct = load_ct_model()
        fusion = None
    else:
        mri = load_mri_model()
        ct = load_ct_model()
        fusion = load_fusion_model()
    return mri, ct, fusion


@st.cache_resource
def load_mri_stage2_model():
    """Load the MRI Stage 2 multiclass model."""
    if not os.path.exists(STAGE2_MODEL_PATH):
        st.warning(f"MRI Stage 2 model not found at {STAGE2_MODEL_PATH}. Stage 2 predictions disabled.")
        return None
    model = tf.keras.models.load_model(STAGE2_MODEL_PATH)
    return model


def predict_mri_stage2(model, image_batch):
    """Predict tumor subtype from MRI image (3 classes). Returns (class_index, probabilities)."""
    if model is None:
        return None, None
    probs = model.predict(image_batch, verbose=0)[0]
    class_idx = np.argmax(probs)
    return class_idx, probs


# ----------------------------------------------------------------------
# Performance Display Functions (using saved images)
# ----------------------------------------------------------------------
def show_mri_performance():
    """Display saved MRI performance figures."""
    fig_dir = './Results/MRI_figures_two_stage'
    st.subheader("Stage 1: Normal vs Tumor")
    cm_img = os.path.join(fig_dir, 'stage1_confusion_matrix.png')
    roc_img = os.path.join(fig_dir, 'stage1_roc_analysis.png')
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Stage 1 Confusion Matrix", use_container_width=True)
    else:
        st.info("Stage 1 confusion matrix image not found.")
    if os.path.exists(roc_img):
        st.image(roc_img, caption="Stage 1 ROC Curve", use_container_width=True)
    else:
        st.info("Stage 1 ROC curve image not found.")

    st.markdown("---")
    st.subheader("Stage 2: Tumor Subtyping (Meningioma, Glioma, Pituitary)")
    cm2_img = os.path.join(fig_dir, 'stage2_confusion_matrix.png')
    roc2_img = os.path.join(fig_dir, 'stage2_roc_curves.png')
    if os.path.exists(cm2_img):
        st.image(cm2_img, caption="Stage 2 Confusion Matrix", use_container_width=True)
    else:
        st.info("Stage 2 confusion matrix image not found.")
    if os.path.exists(roc2_img):
        st.image(roc2_img, caption="Stage 2 ROC Curves", use_container_width=True)
    else:
        st.info("Stage 2 ROC curves image not found.")


def show_ct_performance():
    """Display saved CT performance figures."""
    fig_dir = './Results/CT_figures'
    cm_img = os.path.join(fig_dir, 'ct_confusion_matrix.png')
    roc_img = os.path.join(fig_dir, 'ct_roc_curve.png')
    if os.path.exists(cm_img):
        st.image(cm_img, caption="CT Confusion Matrix", use_container_width=True)
    else:
        st.info("CT confusion matrix image not found.")
    if os.path.exists(roc_img):
        st.image(roc_img, caption="CT ROC Curve", use_container_width=True)
    else:
        st.info("CT ROC curve image not found.")


def show_fusion_performance():
    """Display saved Fusion performance figures."""
    fig_dir = './Results/Fusion_figures'
    cm_img = os.path.join(fig_dir, 'fusion_confusion_matrix.png')
    roc_img = os.path.join(fig_dir, 'fusion_roc_curve.png')
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Fusion Confusion Matrix", use_container_width=True)
    else:
        st.info("Fusion confusion matrix image not found.")
    if os.path.exists(roc_img):
        st.image(roc_img, caption="Fusion ROC Curve", use_container_width=True)
    else:
        st.info("Fusion ROC curve image not found.")


# ----------------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
    st.title("🧠 Brain Tumor Detection with Red Circle Localization")
    st.markdown("Upload MRI and/or CT images. If a tumor is detected, a **red circle** will highlight the suspicious region on the **preprocessed image** (first channel).")

    with st.sidebar:
        modality = st.radio(
            "Select Modality",
            ("MRI only", "CT only", "Both (MRI + CT)"),
            index=0,
        )
        st.markdown("---")
        st.info("Models are loaded once and cached for faster subsequent runs.")

    with st.spinner("Loading models..."):
        model_mri, model_ct, model_fusion = load_models(modality)
        # Load Stage 2 model for MRI only (if needed)
        if modality == "MRI only":
            model_mri_stage2 = load_mri_stage2_model()
        else:
            model_mri_stage2 = None

    pipeline = MultimodalBrainTumorPreprocessingPipeline(target_size=(224, 224))

    # ------------------------------------------------------------------
    # MRI only (with Stage 2 subtyping)
    # ------------------------------------------------------------------
    if modality == "MRI only":
        uploaded_file = st.file_uploader(
            "Upload MRI image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("Preprocessing and predicting..."):
                    batch = preprocess_single_image(tmp_path, "MRI", pipeline)
                    if batch is None:
                        st.error("Preprocessing failed.")
                    else:
                        pred_class, prob = predict_mri(model_mri, batch)

                        # Display original image (grayscale) and preprocessed first channel
                        col1, col2 = st.columns(2)
                        orig = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
                        if orig is not None:
                            col1.image(orig, caption="Original Image", width='stretch')
                        disp_img = get_display_image(batch, "MRI")
                        col2.image(disp_img, caption="Preprocessed (first channel)", width='stretch')

                        st.success(f"**Prediction:** {CLASS_NAMES[pred_class]} (confidence: {prob:.2%})")

                        # Stage 2: if tumor detected and Stage 2 model is available
                        if pred_class == 1 and model_mri_stage2 is not None:
                            with st.spinner("Performing tumor subtyping..."):
                                try:
                                    subtype_idx, subtype_probs = predict_mri_stage2(model_mri_stage2, batch)
                                    subtype_name = STAGE2_CLASS_NAMES[subtype_idx]
                                    st.info(f"**Tumor Subtype:** {subtype_name} (confidence: {subtype_probs[subtype_idx]:.2%})")
                                except Exception as e:
                                    st.warning(f"Subtyping failed: {e}")
                        elif pred_class == 1 and model_mri_stage2 is None:
                            st.warning("Stage 2 model not loaded – tumor subtyping not available.")

                        if pred_class == 1:
                            with st.spinner("Locating tumor region..."):
                                try:
                                    heatmap = generate_occlusion_heatmap(model_mri, batch, pred_class)
                                    overlay = draw_tumor_circle(disp_img, heatmap)
                                    st.image(overlay, caption="Tumor Localization (red circle on preprocessed image)", width='stretch')
                                except Exception as e:
                                    st.warning(f"Localization could not be generated: {e}")
                        else:
                            st.info("No tumor detected – localization not performed.")
            finally:
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # CT only
    # ------------------------------------------------------------------
    elif modality == "CT only":
        uploaded_file = st.file_uploader(
            "Upload CT image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("Preprocessing and predicting..."):
                    batch = preprocess_single_image(tmp_path, "CT", pipeline)
                    if batch is None:
                        st.error("Preprocessing failed.")
                    else:
                        pred_class, prob = predict_ct(model_ct, batch)

                        col1, col2 = st.columns(2)
                        orig = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
                        if orig is not None:
                            col1.image(orig, caption="Original Image", width='stretch')
                        disp_img = get_display_image(batch, "CT")
                        col2.image(disp_img, caption="Preprocessed (first channel)", width='stretch')

                        st.success(f"**Prediction:** {CLASS_NAMES[pred_class]} (confidence: {prob:.2%})")

                        if pred_class == 1:
                            with st.spinner("Locating tumor region..."):
                                try:
                                    heatmap = generate_occlusion_heatmap(model_ct, batch, pred_class)
                                    overlay = draw_tumor_circle(disp_img, heatmap)
                                    st.image(overlay, caption="Tumor Localization (red circle on preprocessed image)", width='stretch')
                                except Exception as e:
                                    st.warning(f"Localization could not be generated: {e}")
                        else:
                            st.info("No tumor detected – localization not performed.")
            finally:
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Fusion (Both)
    # ------------------------------------------------------------------
    else:
        st.subheader("Upload paired images (MRI and CT)")
        mri_file = st.file_uploader("MRI image", type=["png","jpg","jpeg","bmp","tif","tiff"], key="mri")
        ct_file = st.file_uploader("CT image", type=["png","jpg","jpeg","bmp","tif","tiff"], key="ct")

        if mri_file and ct_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(mri_file.name).suffix) as tmp_mri:
                tmp_mri.write(mri_file.getbuffer())
                mri_path = tmp_mri.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(ct_file.name).suffix) as tmp_ct:
                tmp_ct.write(ct_file.getbuffer())
                ct_path = tmp_ct.name

            try:
                with st.spinner("Preprocessing and predicting..."):
                    mri_batch = preprocess_single_image(mri_path, "MRI", pipeline)
                    ct_batch = preprocess_single_image(ct_path, "CT", pipeline)
                    if mri_batch is None or ct_batch is None:
                        st.error("Preprocessing failed for one or both images.")
                    else:
                        pred_class, prob = predict_fusion(model_fusion, mri_batch, ct_batch)

                        # Display original images and preprocessed first channels
                        col1, col2 = st.columns(2)
                        orig_mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
                        if orig_mri is not None:
                            col1.image(orig_mri, caption="MRI Original", width='stretch')
                        disp_mri = get_display_image(mri_batch, "MRI")
                        col2.image(disp_mri, caption="MRI Preprocessed", width='stretch')

                        col3, col4 = st.columns(2)
                        orig_ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
                        if orig_ct is not None:
                            col3.image(orig_ct, caption="CT Original", width='stretch')
                        disp_ct = get_display_image(ct_batch, "CT")
                        col4.image(disp_ct, caption="CT Preprocessed", width='stretch')

                        st.success(f"**Prediction:** {CLASS_NAMES[pred_class]} (confidence: {prob:.2%})")

                        if pred_class == 1:
                            with st.spinner("Locating tumor region on MRI..."):
                                try:
                                    heatmap = generate_occlusion_heatmap(model_mri, mri_batch, pred_class)
                                    overlay = draw_tumor_circle(disp_mri, heatmap)
                                    st.image(overlay, caption="Tumor Localization on MRI (red circle)", width='stretch')
                                except Exception as e:
                                    st.warning(f"Localization could not be generated: {e}")
                        else:
                            st.info("No tumor detected – localization not performed.")
            finally:
                os.unlink(mri_path)
                os.unlink(ct_path)

    # ------------------------------------------------------------------
    # Batch Processing (multi‑file upload)
    # ------------------------------------------------------------------
    with st.expander("Batch Processing (upload multiple files)"):
        st.markdown("Upload multiple MRI or CT images (for single modality only).")
        if modality == "Both (MRI + CT)":
            st.warning("Batch processing is only supported for MRI only or CT only.")
        else:
            uploaded_files = st.file_uploader(
                "Choose images",
                type=["png","jpg","jpeg","bmp","tif","tiff"],
                accept_multiple_files=True
            )
            if uploaded_files and st.button("Process Batch"):
                progress_bar = st.progress(0)
                results = []
                mod = modality.split()[0]  # "MRI" or "CT"
                total = len(uploaded_files)
                for i, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        batch = preprocess_single_image(tmp_path, mod, pipeline)
                        if batch is not None:
                            if mod == "MRI":
                                pred_class, prob = predict_mri(model_mri, batch)
                                # For MRI, optionally run Stage 2
                                if pred_class == 1 and model_mri_stage2 is not None:
                                    subtype_idx, _ = predict_mri_stage2(model_mri_stage2, batch)
                                    subtype = STAGE2_CLASS_NAMES[subtype_idx]
                                    results.append((uploaded_file.name, f"Tumor ({subtype})", prob))
                                else:
                                    results.append((uploaded_file.name, CLASS_NAMES[pred_class], prob))
                            else:
                                pred_class, prob = predict_ct(model_ct, batch)
                                results.append((uploaded_file.name, CLASS_NAMES[pred_class], prob))
                        else:
                            results.append((uploaded_file.name, "Error", 0.0))
                    except Exception as e:
                        results.append((uploaded_file.name, f"Error: {e}", 0.0))
                    finally:
                        os.unlink(tmp_path)
                    progress_bar.progress((i+1)/total)

                # Display results table
                df = pd.DataFrame(results, columns=["Filename", "Prediction", "Tumor Probability"])
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "results.csv", "text/csv")

    # ------------------------------------------------------------------
    # Model Info & Performance Graphs
    # ------------------------------------------------------------------
    with st.expander("Model Info & Performance"):
        st.write("Model files used:")
        st.code(f"""
        MRI Stage 1:  {MODEL_PATHS['mri_stage1']}
        MRI Stage 2:  {STAGE2_MODEL_PATH}
        CT:           {MODEL_PATHS['ct_correlation']}
        Fusion:       {MODEL_PATHS['fusion']}
        """)

        st.markdown("---")
        st.subheader("Pre‑computed Model Performance (from training)")

        # Show performance graphs for each modality
        tab1, tab2, tab3 = st.tabs(["MRI", "CT", "Fusion"])
        with tab1:
            show_mri_performance()
        with tab2:
            show_ct_performance()
        with tab3:
            show_fusion_performance()


if __name__ == "__main__":
    main()