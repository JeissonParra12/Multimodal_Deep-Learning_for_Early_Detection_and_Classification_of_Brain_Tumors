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
from PIL import Image

# Import existing modules
from Preprocessing_for_prediction import MultimodalBrainTumorPreprocessingPipeline
from Preprocessing_for_implementation import (
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
    CorrelationLayer,
    MODEL_PATHS,
    find_last_conv_layer,
)
gradcam_weight=0.35
occlusion_weight=0.65
# Define additional model paths (add Stage 2)
STAGE2_MODEL_PATH = './Saved_models/MRI_stage2_multiclass_model.keras'
STAGE2_CLASS_NAMES = ['Meningioma', 'Glioma', 'Pituitary']

st.warning(
"""
⚠️ Disclaimer: This tool is intended strictly for research and educational purposes and does not constitute a medical device. 
Predictions are not for clinical diagnosis or treatment decisions and must be interpreted by qualified healthcare professionals.
"""
)
# ----------------------------------------------------------------------
# Occlusion‑based Localization and Grad-CAM Functions
# ----------------------------------------------------------------------
def generate_gradcam_heatmap(model, image_batch, class_index=None, layer_name=None):
    """
    Generate Grad-CAM heatmap for a single input batch.
    Returns heatmap normalized to [0,1].
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)

    # Average gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # shape: (1, channels)

    conv_outputs = conv_outputs[0]      # (H, W, C)
    pooled_grads = pooled_grads[0]      # (C,)

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

def overlay_heatmap_on_image(image, heatmap, alpha=0.40):
    """
    Overlay Grad-CAM heatmap on grayscale image.
    image: uint8 grayscale or BGR
    heatmap: float [0,1]
    """
    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
    return overlay

def apply_brain_mask_to_heatmap(display_img, heatmap):
    """
    Restrict heatmap to the visible brain region only.
    display_img: uint8 grayscale image
    heatmap: float [0,1]
    """
    heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))

    # Rough brain mask from non-black pixels
    _, brain_mask = cv2.threshold(display_img, 10, 1, cv2.THRESH_BINARY)
    brain_mask = brain_mask.astype(np.float32)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)

    masked_heatmap = heatmap_resized * brain_mask

    if masked_heatmap.max() > 0:
        masked_heatmap = masked_heatmap / masked_heatmap.max()

    return masked_heatmap
def normalize_heatmap(heatmap):
    heatmap = heatmap.astype(np.float32)
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap
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

def generate_hybrid_heatmap(model, image_batch, display_img, class_index,
                            gradcam_weight=0.35, occlusion_weight=0.65,
                            patch_size=20, stride=10):
    """
    Hybrid explanation:
    - Grad-CAM gives smooth global attention
    - Occlusion gives more faithful local importance

    Returns normalized hybrid heatmap in display_img space.
    """

    # Grad-CAM
    gradcam = generate_gradcam_heatmap(model, image_batch, class_index=class_index)
    gradcam = cv2.resize(gradcam, (display_img.shape[1], display_img.shape[0]))
    gradcam = normalize_heatmap(gradcam)

    # Occlusion
    occlusion = generate_occlusion_heatmap(
        model,
        image_batch,
        class_index,
        patch_size=patch_size,
        stride=stride
    )
    occlusion = cv2.resize(occlusion, (display_img.shape[1], display_img.shape[0]))
    occlusion = normalize_heatmap(occlusion)

    # Brain masking
    gradcam = apply_brain_mask_to_heatmap(display_img, gradcam)
    occlusion = apply_brain_mask_to_heatmap(display_img, occlusion)

    # Weighted fusion
    hybrid = gradcam_weight * gradcam + occlusion_weight * occlusion
    hybrid = normalize_heatmap(hybrid)

    # Light smoothing for cleaner medical-style display
    hybrid = cv2.GaussianBlur(hybrid, (7, 7), 0)
    hybrid = normalize_heatmap(hybrid)

    return hybrid, gradcam, occlusion

def draw_tumor_circle(image, heatmap, min_radius=8, max_radius=45):
    """
    Draw circle around the connected component containing the hottest point.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_blur = cv2.GaussianBlur(heatmap_resized, (9, 9), 0)

    positive_vals = heatmap_blur[heatmap_blur > 0]
    if len(positive_vals) == 0:
        return image.copy()

    # Use a high percentile to isolate strongest activation
    threshold = np.percentile(positive_vals, 92)
    mask = (heatmap_blur >= threshold).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    overlay = image.copy()

    # Find hottest point
    _, _, _, maxLoc = cv2.minMaxLoc(heatmap_blur)
    max_x, max_y = maxLoc

    selected_label = labels[max_y, max_x]

    if selected_label > 0:
        x = stats[selected_label, cv2.CC_STAT_LEFT]
        y = stats[selected_label, cv2.CC_STAT_TOP]
        ww = stats[selected_label, cv2.CC_STAT_WIDTH]
        hh = stats[selected_label, cv2.CC_STAT_HEIGHT]

        cx = int(centroids[selected_label][0])
        cy = int(centroids[selected_label][1])

        radius = int(max(ww, hh) / 2)
        radius = max(min_radius, min(radius, max_radius))

        cv2.circle(overlay, (cx, cy), radius, (255, 0, 0), thickness=3)

    return overlay

def get_display_image(processed_batch, modality):
    """Extract first channel of preprocessed 4-channel array for display."""
    img = processed_batch[0]
    display = (img[:, :, 0] * 255).astype(np.uint8)
    return display


# ----------------------------------------------------------------------
# Model Loading with Caching
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
def load_mri_stage2_model():
    """Load the MRI Stage 2 multiclass model."""
    if not os.path.exists(STAGE2_MODEL_PATH):
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
    st.markdown("#### Stage 1: Normal vs Tumor")
    cm_img = os.path.join(fig_dir, 'stage1_confusion_matrix.png')
    roc_img = os.path.join(fig_dir, 'stage1_roc_analysis.png')
    
    col1, col2 = st.columns(2)
    if os.path.exists(cm_img):
        col1.image(cm_img, caption="Stage 1 Confusion Matrix", use_container_width=True)
    else:
        col1.info("Stage 1 confusion matrix image not found.")
    if os.path.exists(roc_img):
        col2.image(roc_img, caption="Stage 1 ROC Curve", use_container_width=True)
    else:
        col2.info("Stage 1 ROC curve image not found.")

    st.markdown("---")
    st.markdown("#### Stage 2: Tumor Subtyping (Meningioma, Glioma, Pituitary)")
    cm2_img = os.path.join(fig_dir, 'stage2_confusion_matrix.png')
    roc2_img = os.path.join(fig_dir, 'stage2_roc_curves.png')
    
    col3, col4 = st.columns(2)
    if os.path.exists(cm2_img):
        col3.image(cm2_img, caption="Stage 2 Confusion Matrix", use_container_width=True)
    else:
        col3.info("Stage 2 confusion matrix image not found.")
    if os.path.exists(roc2_img):
        col4.image(roc2_img, caption="Stage 2 ROC Curves", use_container_width=True)
    else:
        col4.info("Stage 2 ROC curves image not found.")


def show_ct_performance():
    """Display saved CT performance figures."""
    fig_dir = './Results/CT_figures'
    cm_img = os.path.join(fig_dir, 'ct_confusion_matrix.png')
    roc_img = os.path.join(fig_dir, 'ct_roc_curve.png')
    
    col1, col2 = st.columns(2)
    if os.path.exists(cm_img):
        col1.image(cm_img, caption="CT Confusion Matrix", use_container_width=True)
    else:
        col1.info("CT confusion matrix image not found.")
    if os.path.exists(roc_img):
        col2.image(roc_img, caption="CT ROC Curve", use_container_width=True)
    else:
        col2.info("CT ROC curve image not found.")


def show_fusion_performance():
    """Display saved Fusion performance figures."""
    fig_dir = './Results/Fusion_figures'
    cm_img = os.path.join(fig_dir, 'fusion_confusion_matrix.png')
    roc_img = os.path.join(fig_dir, 'fusion_roc_curve.png')
    
    col1, col2 = st.columns(2)
    if os.path.exists(cm_img):
        col1.image(cm_img, caption="Fusion Confusion Matrix", use_container_width=True)
    else:
        col1.info("Fusion confusion matrix image not found.")
    if os.path.exists(roc_img):
        col2.image(roc_img, caption="Fusion ROC Curve", use_container_width=True)
    else:
        col2.info("Fusion ROC curve image not found.")

# ----------------------------------------------------------------------
# CSS Styling
# ----------------------------------------------------------------------
def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f4f8;
            color: #0f172a;
        }

        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #0f172a;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1e3a8a !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] li {
            color: #0f172a !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            color: #0f172a !important;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #e0f2fe !important;
            border-bottom: 3px solid #0284c7;
            color: #0369a1 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            color: #0f172a !important;
        }

        [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        header[data-testid="stHeader"] {
            background-color: #f8fafc !important;
        }

        button[kind="header"],
        [data-testid="collapsedControl"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }

        button[kind="header"] svg,
        [data-testid="collapsedControl"] svg {
            fill: #0f172a !important;
            color: #0f172a !important;
        }

        .stRadio label, .stSelectbox label, .stFileUploader label {
            color: #0f172a !important;
        }

        [data-testid="stFileUploadDropzone"] {
            border: 2px dashed #94a3b8 !important;
            background-color: #f8fafc !important;
            border-radius: 10px;
            color: #0f172a !important;
        }

        [data-testid="stFileUploadDropzone"] * {
            color: #0f172a !important;
        }

        [data-testid="stFileUploadDropzoneInstructions"] {
            color: #0f172a !important;
        }

        [data-testid="stFileUploader"] * {
            color: #0f172a !important;
        }

        .stButton > button {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }

        .sidebar-info-box {
            background-color: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-size: 0.9em;
            color: #334155 !important;
            margin-bottom: 20px;
        }

        .sidebar-info-box * {
            color: #334155 !important;
        }

        .result-box-normal {
            background-color: #d1fae5;
            border-left: 5px solid #10b981;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: #065f46 !important;
        }

        .result-box-tumor {
            background-color: #fee2e2;
            border-left: 5px solid #ef4444;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: #991b1b !important;
        }

        [data-testid="stAlertContainer"] * {
            color: #0f172a !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
    apply_custom_css()
    
    st.title("🧠 Brain Tumor Detection")
    st.markdown("""
    <span style="color: #475569; font-size: 1.1em;">
    Multimodal diagnostic tool for MRI and CT scans.
    </span>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <span style="color: #475569; font-size: 1.1em;">
    Upload imaging data to perform automated tumor classification, clinical subtyping, and automated lesion localization.
    </span>
    """, unsafe_allow_html=True)
    st.write("") # Spacer

    with st.sidebar:
        #  FIU Logo and Project Info
        try:
            st.image("Results/FIU.png", use_container_width=True)
        except Exception:
            # Fallback if the image path is incorrect
            st.warning("FIU logo not found at 'Results/FIU.png'")

        st.markdown("""
        <div class="sidebar-info-box">
            <h4 style="margin-top: 0; color: #1e3a8a;">Capstone Project</h4>
            <strong>Multimodal Deep Learning for Early Detection and Classification of Brain Tumors Using MRI and CT Scans</strong><br><br>
             <strong>Author:</strong> Jeisson Farid Parra Prieto<br>
             <strong>Course:</strong> IDC 6940 Capstone in Data Science<br>
             <strong>Mentor:</strong> Dr. Fahad Saeed<br>
             <strong>Instructor:</strong> Dr. Ananda M. Mondal
        </div>
        """, unsafe_allow_html=True)
        # ---------------------------------------

        st.markdown("### Clinical Settings")
        modality = st.radio(
            "Select Imaging Modality",
            ("MRI only", "CT only", "Both (MRI + CT)"),
            index=0,
            help="Choose the type of scan you are uploading. Fusion (Both) yields the highest accuracy if paired scans are available."
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        with st.spinner("Initializing AI Models..."):
            model_mri, model_ct, model_fusion = load_models(modality)
            if modality == "MRI only":
                model_mri_stage2 = load_mri_stage2_model()
                if model_mri_stage2 is None:
                    st.warning("⚠️ Stage 2 model missing.")
            else:
                model_mri_stage2 = None
        st.success("✅ Models Loaded & Cached")
        st.caption("Powered by TensorFlow & Keras")

    pipeline = MultimodalBrainTumorPreprocessingPipeline(target_size=(224, 224))

    # --- MAIN TABS ---
    tab_single, tab_batch, tab_perf = st.tabs([
        "🔍 Single Scan Analysis", 
        "📂 Scan Multiple Images", 
        "📊 Model Performance & Architecture"
    ])

    # ==================================================================
    # TAB 1: SINGLE SCAN ANALYSIS
    # ==================================================================
    with tab_single:
        st.markdown("### Upload Medical Imaging")
        
        # Helper dictionary to manage allowed files
        allowed_types = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        
        # ------------------- MRI ONLY -------------------
        if modality == "MRI only":
            uploaded_file = st.file_uploader(
                "Upload a high-resolution MRI scan",
                type=allowed_types,
                help="Supported formats: PNG, JPG, BMP, TIFF"
            )
            
            if uploaded_file is not None:
                # Layout for results
                st.markdown("---")
                st.markdown("### Diagnostic Results")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    with st.spinner("Analyzing scan parameters and extracting features..."):
                        batch = preprocess_single_image(tmp_path, "MRI", pipeline)
                        
                        if batch is None:
                            st.error("❌ Preprocessing failed. The image format might be corrupted or unsupported.")
                        else:
                            pred_class, prob = predict_mri(model_mri, batch)
                            
                            # Determine class formatting
                            if pred_class == 1:
                                st.markdown(f"""
                                <div class="result-box-tumor">
                                    st.caption("Results are for research purposes only and not for medical diagnosis.")
                                    <strong>Primary Detection:</strong> Tumor Detected (Confidence: {prob:.2%})
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box-normal">
                                    <strong>Primary Detection:</strong> Normal Scan (Confidence: {(1-prob):.2%})
                                </div>
                                """, unsafe_allow_html=True)

                            # Stage 2 Subtyping
                            if pred_class == 1 and model_mri_stage2 is not None:
                                with st.spinner("Performing clinical subtyping..."):
                                    try:
                                        subtype_idx, subtype_probs = predict_mri_stage2(model_mri_stage2, batch)
                                        subtype_name = STAGE2_CLASS_NAMES[subtype_idx]
                                        st.info(f"🧬 **Tumor Subtype Identified:** {subtype_name} (Confidence: {subtype_probs[subtype_idx]:.2%})")
                                    except Exception as e:
                                        st.warning(f"Subtyping failed: {e}")

                            # Image Visualization Columns
                            st.markdown("#### Imaging Visualizations")
                            col1, col2, col3, col4 = st.columns(4)

                            orig = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
                            if orig is not None:
                                col1.image(orig, caption="Original Scan", use_container_width=True)

                            disp_img = get_display_image(batch, "MRI")
                            col2.image(disp_img, caption="Preprocessed MRI", use_container_width=True)

                            if pred_class == 1:
                                with st.spinner("Generating Grad-CAM visualization..."):
                                    try:
                                        hybrid_heatmap, gradcam_map, occlusion_map = generate_hybrid_heatmap(
                                            model_mri,
                                            batch,
                                            disp_img,
                                            class_index=pred_class,
                                            gradcam_weight=0.35,
                                            occlusion_weight=0.65,
                                            patch_size=20,
                                            stride=10
                                        )

                                        heatmap_overlay = overlay_heatmap_on_image(disp_img, hybrid_heatmap, alpha=0.40)
                                        circle_overlay = draw_tumor_circle(disp_img, hybrid_heatmap)

                                        col3.image(heatmap_overlay, caption="Hybrid Heatmap", use_container_width=True)
                                        col4.image(circle_overlay, caption="Hybrid Lesion Localization", use_container_width=True)
                                    except Exception as e:
                                        col3.warning(f"Heatmap failed: {e}")
                                        col4.warning("Localization unavailable")
                            else:
                                st.caption("Results are for research purposes only and not for medical diagnosis.")
                                col3.info("No tumor detected")
                                col4.info("No localization required")
                finally:
                    os.unlink(tmp_path)

        # ------------------- CT ONLY -------------------
        elif modality == "CT only":
            uploaded_file = st.file_uploader(
                "Upload a high-resolution CT scan",
                type=allowed_types,
                help="Supported formats: PNG, JPG, BMP, TIFF"
            )
            
            if uploaded_file is not None:
                st.markdown("---")
                st.markdown("### Diagnostic Results")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    with st.spinner("Analyzing scan parameters and extracting features..."):
                        batch = preprocess_single_image(tmp_path, "CT", pipeline)
                        if batch is None:
                            st.error("❌ Preprocessing failed. The image format might be corrupted or unsupported.")
                        else:
                            pred_class, prob = predict_ct(model_ct, batch)

                            if pred_class == 1:
                                st.markdown(f"""
                                <div class="result-box-tumor">
                                    st.caption("Results are for research purposes only and not for medical diagnosis.")
                                    <strong>Primary Detection:</strong> Tumor Detected (Confidence: {prob:.2%})
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box-normal">
                                    st.caption("Results are for research purposes only and not for medical diagnosis.")
                                    <strong>Primary Detection:</strong> Normal Scan (Confidence: {(1-prob):.2%})
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("#### Imaging Visualizations")
                            col1, col2, col3 = st.columns(3)
                            
                            orig = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
                            if orig is not None:
                                col1.image(orig, caption="Original Scan", use_container_width=True)
                                
                            disp_img = get_display_image(batch, "CT")
                            col2.image(disp_img, caption="Preprocessed Scan", use_container_width=True)

                            if pred_class == 1:
                                with st.spinner("Generating spatial occlusion maps..."):
                                    try:
                                        heatmap = generate_occlusion_heatmap(model_ct, batch, pred_class)
                                        overlay = draw_tumor_circle(disp_img, heatmap)
                                        col3.image(overlay, caption="Automated Lesion Localization", use_container_width=True)
                                    except Exception as e:
                                        col3.warning(f"Localization unavailable: {e}")
                            else:
                                col3.info("Clear Scan: Localization not required.")
                finally:
                    os.unlink(tmp_path)

        # ------------------- FUSION (BOTH) -------------------
        else:
            st.info("💡 **Fusion Mode:** Please upload both MRI and CT scans for the same patient to leverage the multimodal fusion network.")
            col_up1, col_up2 = st.columns(2)
            
            with col_up1:
                mri_file = st.file_uploader("Upload MRI image", type=allowed_types, key="mri")
            with col_up2:
                ct_file = st.file_uploader("Upload CT image", type=allowed_types, key="ct")

            if mri_file and ct_file:
                st.markdown("---")
                st.markdown("### Multimodal Diagnostic Results")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(mri_file.name).suffix) as tmp_mri:
                    tmp_mri.write(mri_file.getbuffer())
                    mri_path = tmp_mri.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(ct_file.name).suffix) as tmp_ct:
                    tmp_ct.write(ct_file.getbuffer())
                    ct_path = tmp_ct.name

                try:
                    with st.spinner("Cross-referencing multimodal features..."):
                        mri_batch = preprocess_single_image(mri_path, "MRI", pipeline)
                        ct_batch = preprocess_single_image(ct_path, "CT", pipeline)
                        
                        if mri_batch is None or ct_batch is None:
                            st.error("❌ Preprocessing failed for one or both images.")
                        else:
                            pred_class, prob = predict_fusion(model_fusion, mri_batch, ct_batch)

                            if pred_class == 1:
                                st.markdown(f"""
                                <div class="result-box-tumor">
                                    st.caption("Results are for research purposes only and not for medical diagnosis.")
                                    <strong>Multimodal Detection:</strong> Tumor Detected (Fusion Confidence: {prob:.2%})
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box-normal">
                                    st.caption("Results are for research purposes only and not for medical diagnosis.")
                                    <strong>Multimodal Detection:</strong> Normal Scans (Fusion Confidence: {(1-prob):.2%})
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("#### Fusion Imaging Visualizations")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            orig_mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
                            if orig_mri is not None:
                                col1.image(orig_mri, caption="MRI Original", use_container_width=True)
                            disp_mri = get_display_image(mri_batch, "MRI")
                            col2.image(disp_mri, caption="MRI Preprocessed", use_container_width=True)

                            orig_ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
                            if orig_ct is not None:
                                col3.image(orig_ct, caption="CT Original", use_container_width=True)
                            disp_ct = get_display_image(ct_batch, "CT")
                            col4.image(disp_ct, caption="CT Preprocessed", use_container_width=True)

                            if pred_class == 1:
                                with st.spinner("Locating tumor region via MRI alignment..."):
                                    try:
                                        # Use MRI model as proxy for localization display since it maps visually better
                                        heatmap = generate_occlusion_heatmap(model_mri, mri_batch, pred_class)
                                        overlay = draw_tumor_circle(disp_mri, heatmap)
                                        
                                        st.markdown("#### 🎯 Multimodal Lesion Localization")
                                        st.image(overlay, caption="Tumor Localization on MRI Base", width=300)
                                    except Exception as e:
                                        st.warning(f"Localization could not be generated: {e}")
                finally:
                    os.unlink(mri_path)
                    os.unlink(ct_path)


    # ==================================================================
    # TAB 2: BATCH PROCESSING
    # ==================================================================
    with tab_batch:
        st.markdown("### 📂 Upload Multiple Files")
        st.write("Upload a cohort of imaging files for rapid, automated triage.")
        
        if modality == "Both (MRI + CT)":
            st.warning("⚠️ Batch processing is currently only supported for standalone MRI or standalone CT modes. Please switch modality in the sidebar.")
        else:
            uploaded_files = st.file_uploader(
                f"Select multiple {modality.split()[0]} scans",
                type=allowed_types,
                accept_multiple_files=True,
                help="You can drag and drop multiple files here."
            )
            
            if uploaded_files and st.button("🚀 Process ", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                mod = modality.split()[0]  # "MRI" or "CT"
                total = len(uploaded_files)
                
                # Container for the visual gallery
                st.markdown("### Processing Gallery")
                gallery_container = st.container()
                cols = gallery_container.columns(4)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {i+1} of {total}...")
                    
                    # Read image for the thumbnail
                    image_bytes = uploaded_file.getvalue()
                    try:
                        thumb_img = Image.open(uploaded_file).convert('RGB')
                        thumb_img.thumbnail((200, 200))
                    except Exception:
                        thumb_img = None

                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name
                        
                    try:
                        batch = preprocess_single_image(tmp_path, mod, pipeline)
                        if batch is not None:
                            if mod == "MRI":
                                pred_class, prob = predict_mri(model_mri, batch)
                                if pred_class == 1 and model_mri_stage2 is not None:
                                    subtype_idx, _ = predict_mri_stage2(model_mri_stage2, batch)
                                    subtype = STAGE2_CLASS_NAMES[subtype_idx]
                                    prediction_label = f"Tumor ({subtype})"
                                else:
                                    prediction_label = CLASS_NAMES[pred_class]
                            else:
                                pred_class, prob = predict_ct(model_ct, batch)
                                prediction_label = CLASS_NAMES[pred_class]
                                
                            results.append({"name": uploaded_file.name, "pred": prediction_label, "prob": prob, "thumb": thumb_img})
                        else:
                            results.append({"name": uploaded_file.name, "pred": "Error", "prob": 0.0, "thumb": thumb_img})
                    except Exception as e:
                        results.append({"name": uploaded_file.name, "pred": f"Error: {e}", "prob": 0.0, "thumb": thumb_img})
                    finally:
                        os.unlink(tmp_path)
                        
                    # Update Gallery immediately
                    current_res = results[-1]
                    with cols[i % 4]:
                        st.markdown(f"<div style='border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                        if current_res['thumb']:
                            st.image(current_res['thumb'], use_container_width=True)
                        st.markdown(f"**{current_res['name'][:15]}...**", help=current_res['name'])
                        
                        if "Error" in current_res['pred']:
                            st.error(current_res['pred'])
                        elif "Tumor" in current_res['pred']:
                            st.error(f"⚠️ {current_res['pred']} ({current_res['prob']:.1%})")
                        else:
                            st.success(f"✅ {current_res['pred']} ({1 - current_res['prob']:.1%})")
                        st.markdown("</div>", unsafe_allow_html=True)

                    progress_bar.progress((i+1)/total)

                status_text.success("Batch Processing Complete!")
                
                # Tabular Summary & Download
                st.markdown("### Export Summary")
                df_results = pd.DataFrame([
                    {"Filename": r['name'], "Prediction": r['pred'], "Confidence of tumor": f"{r['prob']:.4f}"} 
                    for r in results
                ])
                st.dataframe(df_results, use_container_width=True)
                
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Report", 
                    data=csv, 
                    file_name="clinical_batch_results.csv", 
                    mime="text/csv"
                )

    # ==================================================================
    # TAB 3: MODEL PERFORMANCE
    # ==================================================================
    with tab_perf:
        st.markdown("### 📊 Network Architecture & Pre-computed Metrics")
        st.write("Review the cross-validated performance metrics and internal weights paths utilized by the system.")
        
        with st.expander("Show Internal Architecture Paths"):
            st.code(f"""
            MRI Stage 1:  {MODEL_PATHS['mri_stage1']}
            MRI Stage 2:  {STAGE2_MODEL_PATH}
            CT:           {MODEL_PATHS['ct_correlation']}
            Fusion:       {MODEL_PATHS['fusion']}
            """)

        st.markdown("---")
        
        perf_tab1, perf_tab2, perf_tab3 = st.tabs(["🧠 MRI Validation", "🦴 CT Validation", "🧬 Fusion Validation"])
        with perf_tab1:
            show_mri_performance()
        with perf_tab2:
            show_ct_performance()
        with perf_tab3:
            show_fusion_performance()

if __name__ == "__main__":
    main()