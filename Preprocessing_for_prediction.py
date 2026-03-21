import os
import cv2
import numpy as np
from pathlib import Path

class MultimodalBrainTumorPreprocessingPipeline:
    """
    Multimodal preprocessing pipeline for brain tumor MRI and CT images.
    This version is used for inference – it processes a single image path
    and returns the 4‑channel normalized array ready for model input.
    """

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.quality_metrics = {}   # not used during inference, kept for compatibility

    # ------------------------------------------------------------------------
    # Core image I/O
    # ------------------------------------------------------------------------
    def load_image(self, image_path):
        """Load image as grayscale."""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    # ------------------------------------------------------------------------
    # Modality‑specific parameters
    # ------------------------------------------------------------------------
    def get_modality_specific_parameters(self, modality):
        if modality == "MRI":
            return {
                'denoise_h': 12,
                'denoise_strength': 75,
                'clahe_clip_limit': 3.0,
                'gamma_correction': True,
                'brain_extraction': True
            }
        elif modality == "CT":
            return {
                'denoise_h': 8,
                'denoise_strength': 50,
                'clahe_clip_limit': 2.0,
                'gamma_correction': False,
                'brain_extraction': True
            }
        else:
            return {
                'denoise_h': 10,
                'denoise_strength': 60,
                'clahe_clip_limit': 2.5,
                'gamma_correction': True,
                'brain_extraction': True
            }

    # ------------------------------------------------------------------------
    # Brain extraction
    # ------------------------------------------------------------------------
    def extract_brain_region(self, image, modality="MRI"):
        params = self.get_modality_specific_parameters(modality)
        if not params['brain_extraction']:
            return image

        if modality == "CT":
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            result = cv2.bitwise_and(image, image, mask=mask)
            return result
        else:
            return image

    # ------------------------------------------------------------------------
    # Medical denoising
    # ------------------------------------------------------------------------
    def medical_denoise(self, image, modality="MRI"):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        params = self.get_modality_specific_parameters(modality)

        denoised = cv2.fastNlMeansDenoising(
            image,
            h=params['denoise_h'],
            templateWindowSize=7,
            searchWindowSize=21
        )

        denoised = cv2.medianBlur(denoised, 3)
        denoised = cv2.bilateralFilter(denoised, 5,
                                       params['denoise_strength'],
                                       params['denoise_strength'])

        return denoised

    # ------------------------------------------------------------------------
    # Contrast enhancement
    # ------------------------------------------------------------------------
    def advanced_contrast_enhancement(self, image, modality="MRI"):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        params = self.get_modality_specific_parameters(modality)

        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'],
                                 tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        if params['gamma_correction']:
            mean_intensity = np.mean(enhanced)
            gamma = 1.0 - (mean_intensity - 127) / 255 * 0.4
            gamma = max(0.5, min(1.5, gamma))

            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)

        return enhanced

    # ------------------------------------------------------------------------
    # Multi‑scale feature generation
    # ------------------------------------------------------------------------
    def multi_scale_processing(self, image):
        """
        Produce four complementary channels:
        1. original resized to target_size
        2. down‑upsampled version
        3. edge map
        4. texture map
        """
        original = cv2.resize(image, self.target_size)

        # Down‑upsampled
        small = cv2.resize(image, (self.target_size[0]//2, self.target_size[1]//2))
        down_up = cv2.resize(small, self.target_size)

        # Edge map
        edges = cv2.Canny(image, 30, 100)
        edges = cv2.resize(edges, self.target_size)

        # Texture map
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        texture = cv2.resize(texture, self.target_size)

        # Stack into 4 channels
        multi_scale = np.stack([original, down_up, edges, texture], axis=-1)
        return multi_scale

    # ------------------------------------------------------------------------
    # Main preprocessing entry point
    # ------------------------------------------------------------------------
    def modality_specific_preprocessing(self, image_path, modality="MRI"):
        """
        Complete modality‑specific preprocessing for a single image.
        Returns a float32 array of shape (*target_size, 4) normalized to [0,1].
        """
        original_image = self.load_image(image_path)
        if original_image is None:
            return None

        # Apply pipeline steps
        brain_extracted = self.extract_brain_region(original_image, modality)
        denoised = self.medical_denoise(brain_extracted, modality)
        enhanced = self.advanced_contrast_enhancement(denoised, modality)
        processed = self.multi_scale_processing(enhanced)

        # Normalize to [0,1]
        processed = processed.astype(np.float32) / 255.0
        return processed