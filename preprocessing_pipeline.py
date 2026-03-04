import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import shutil

class MultimodalBrainTumorPreprocessingPipeline:
    """
    Multimodal preprocessing pipeline for brain tumor MRI and CT images
    Includes dataset reorganization, modality-specific preprocessing, and class balancing
    """
    
    def __init__(self, target_size=(224, 224), apply_balancing=True):
        self.target_size = target_size
        self.quality_metrics = {}
        self.apply_balancing = apply_balancing
        
        # Define tumor type mappings for filename encoding
        self.tumor_type_mapping = {
            'meningioma_tumor': '1',
            'glioma_tumor': '2', 
            'pituitary_tumor': '3'
        }
    
    # ============================================================================
    # DATASET REORGANIZATION & STANDARDIZATION FUNCTIONS
    # ============================================================================
    
    def list_directory_structure(self, base_dir, max_depth=3):
        """Helper function to list directory structure for debugging"""
        print(f"\nCurrent directory structure of: {base_dir}")
        for root, dirs, files in os.walk(base_dir):
            level = root.replace(base_dir, '').count(os.sep)
            if level <= max_depth:
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files only
                    print(f"{sub_indent}{file}")
                if len(files) > 5:
                    print(f"{sub_indent}... and {len(files) - 5} more files")
    
    def organize_mri_ct_folders(self, base_dir):
        """
        Reorganize MRI and CT folders into standardized structure with tumor-type encoding
        Creates: MRI/normal, MRI/tumor, CT/normal, CT/tumor
        Preserves tumor type information in filenames (1=meningioma, 2=glioma, 3=pituitary)
        """
        
        print(f"Working in base directory: {base_dir}")
        print(f"Base directory exists: {os.path.exists(base_dir)}")
        
        # Create new directory structure
        new_dirs = [
            'MRI/normal',
            'MRI/tumor',
            'CT/normal',
            'CT/tumor'
        ]
        
        for new_dir in new_dirs:
            full_path = os.path.join(base_dir, new_dir)
            os.makedirs(full_path, exist_ok=True)
            print(f"Created directory: {full_path}")
        
        # Counter for tracking file numbers
        file_counters = {
            'MRI_normal': 1,
            'MRI_tumor': 1,
            'CT_normal': 1,
            'CT_tumor': 1
        }
        
        def process_folder(source_path, modality, is_tumor, tumor_type=None):
            """Process individual folders and copy/rename files with tumor-type encoding"""
            
            print(f"Processing folder: {source_path}")
            print(f"Folder exists: {os.path.exists(source_path)}")
            
            if not os.path.exists(source_path):
                print(f"Warning: Folder does not exist: {source_path}")
                return
            
            if modality == 'MRI':
                if is_tumor:
                    category = 'MRI_tumor'
                else:
                    category = 'MRI_normal'
            else:  # CT
                if is_tumor:
                    category = 'CT_tumor'
                else:
                    category = 'CT_normal'
            
            # Process all image files in the folder
            files_found = 0
            for filename in os.listdir(source_path):
                file_path = os.path.join(source_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.dcm')):
                    files_found += 1
                    
                    # Create new filename with tumor-type encoding
                    file_ext = os.path.splitext(filename)[1]
                    
                    if is_tumor and tumor_type and tumor_type in self.tumor_type_mapping:
                        # Include tumor type code in filename
                        new_filename = f"{category}_{file_counters[category]:04d}_{self.tumor_type_mapping[tumor_type]}{file_ext}"
                    else:
                        # Regular filename without tumor type code
                        new_filename = f"{category}_{file_counters[category]:04d}{file_ext}"
                    
                    destination_file = os.path.join(base_dir, modality, 'tumor' if is_tumor else 'normal', new_filename)
                    
                    # Copy file
                    shutil.copy2(file_path, destination_file)
                    print(f"Copied: {filename} -> {new_filename}")
                    
                    file_counters[category] += 1
            
            print(f"Found {files_found} image files in {source_path}")
        
        # Process CT folders
        print("\n" + "="*50)
        
        # CT/kaggle/no_tumor
        ct_normal_path = os.path.join(base_dir, 'CT', 'kaggle', 'no_tumor')
        if os.path.exists(ct_normal_path):
            process_folder(ct_normal_path, 'CT', is_tumor=False)
        else:
            print(f"CT normal path not found: {ct_normal_path}")
        
        # CT/kaggle/tumor
        ct_tumor_path = os.path.join(base_dir, 'CT', 'kaggle', 'tumor')
        if os.path.exists(ct_tumor_path):
            process_folder(ct_tumor_path, 'CT', is_tumor=True)
        else:
            print(f"CT tumor path not found: {ct_tumor_path}")
        
        # Process MRI folders
        print("\n" + "="*50)
        print("Processing MRI folders...")
        
        # MRI/Figshare/1,2,3 (assuming these are tumor types)
        figshare_dirs = ['1', '2', '3']
        for dir_name in figshare_dirs:
            figshare_path = os.path.join(base_dir, 'MRI', 'Figshare', dir_name)
            if os.path.exists(figshare_path):
                tumor_type = None
                if dir_name == '1':
                    tumor_type = 'meningioma_tumor'
                elif dir_name == '2':
                    tumor_type = 'glioma_tumor'
                elif dir_name == '3':
                    tumor_type = 'pituitary_tumor'
                
                process_folder(figshare_path, 'MRI', is_tumor=True, tumor_type=tumor_type)
            else:
                print(f"Figshare path not found: {figshare_path}")
        
        # MRI/Brain 2/no
        brain2_no_path = os.path.join(base_dir, 'MRI', 'Brain 2', 'no')
        if os.path.exists(brain2_no_path):
            process_folder(brain2_no_path, 'MRI', is_tumor=False)
        else:
            print(f"Brain 2 no path not found: {brain2_no_path}")
        
        # MRI/Brain 2/yes
        brain2_yes_path = os.path.join(base_dir, 'MRI', 'Brain 2', 'yes')
        if os.path.exists(brain2_yes_path):
            process_folder(brain2_yes_path, 'MRI', is_tumor=True)
        else:
            print(f"Brain 2 yes path not found: {brain2_yes_path}")
        
        # MRI/Brain Tumor MRI images/Healthy
        brain_tumor_healthy_path = os.path.join(base_dir, 'MRI', 'Brain Tumor MRI images', 'Healthy')
        if os.path.exists(brain_tumor_healthy_path):
            process_folder(brain_tumor_healthy_path, 'MRI', is_tumor=False)
        else:
            print(f"Brain Tumor Healthy path not found: {brain_tumor_healthy_path}")
        
        # MRI/Brain Tumor MRI images/Tumor
        brain_tumor_tumor_path = os.path.join(base_dir, 'MRI', 'Brain Tumor MRI images', 'Tumor')
        if os.path.exists(brain_tumor_tumor_path):
            process_folder(brain_tumor_tumor_path, 'MRI', is_tumor=True)
        else:
            print(f"Brain Tumor Tumor path not found: {brain_tumor_tumor_path}")
        
        # MRI/Brain 1 - Training and Testing folders
        brain1_training_paths = [
            ('no_tumor', False, None),
            ('meningioma_tumor', True, 'meningioma_tumor'),
            ('glioma_tumor', True, 'glioma_tumor'),
            ('pituitary_tumor', True, 'pituitary_tumor')
        ]
        
        for folder_name, is_tumor, tumor_type in brain1_training_paths:
            # Training
            training_path = os.path.join(base_dir, 'MRI', 'Brain 1', 'Training', folder_name)
            if os.path.exists(training_path):
                process_folder(training_path, 'MRI', is_tumor, tumor_type)
            else:
                print(f"Training path not found: {training_path}")
            
            # Testing
            testing_path = os.path.join(base_dir, 'MRI', 'Brain 1', 'Testing', folder_name)
            if os.path.exists(testing_path):
                process_folder(testing_path, 'MRI', is_tumor, tumor_type)
            else:
                print(f"Testing path not found: {testing_path}")
        
        print("\n" + "="*50)
        print("Reorganization completed!")
        print(f"MRI Normal images: {file_counters['MRI_normal'] - 1}")
        print(f"MRI Tumor images: {file_counters['MRI_tumor'] - 1}")
        print(f"CT Normal images: {file_counters['CT_normal'] - 1}")
        print(f"CT Tumor images: {file_counters['CT_tumor'] - 1}")
        
        return file_counters
    
    # ============================================================================
    # IMAGE PREPROCESSING FUNCTIONS
    # ============================================================================
    
    def load_image(self, image_path):
        """Load image with error handling"""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return None
                
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_modality_specific_parameters(self, modality):
        """Return modality-specific preprocessing parameters"""
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
    
    def extract_brain_region(self, image, modality="MRI"):
        """
        Brain region extraction using Otsu thresholding with morphological operations
        Removes skull, background, and non-brain structures
        """
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
    
    def medical_denoise(self, image, modality="MRI"):
        """
        Medical-grade denoising with modality-specific parameters
        MRI: Non-local means denoising (h=12) + median + bilateral filtering
        CT: Non-local means denoising (h=8) + median + bilateral filtering
        """
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
        denoised = cv2.bilateralFilter(denoised, 5, params['denoise_strength'], params['denoise_strength'])
        
        return denoised
    
    def advanced_contrast_enhancement(self, image, modality="MRI"):
        """
        Multi-stage contrast enhancement with modality-specific adjustments
        Uses CLAHE with modality-specific clip limits + adaptive gamma correction for MRI
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        params = self.get_modality_specific_parameters(modality)
            
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'], tileGridSize=(8, 8))
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
    
    def multi_scale_processing(self, image):
        """
        Multi-scale feature preservation generating four complementary channels:
        1. Original resized (224×224)
        2. Downsampled-upsampled (global context)
        3. Edge maps (Canny edge detection)
        4. Texture maps (Sobel gradient magnitude)
        """
        original = cv2.resize(image, self.target_size)
        downsampled = cv2.resize(image, (self.target_size[0]//2, self.target_size[1]//2))
        downsampled = cv2.resize(downsampled, self.target_size)
        edges = cv2.Canny(image, 30, 100)
        edges = cv2.resize(edges, self.target_size)
        
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        texture = cv2.resize(texture, self.target_size)
        
        multi_scale = np.stack([original, downsampled, edges, texture], axis=-1)
        return multi_scale
    
    def modality_specific_preprocessing(self, image_path, modality="MRI", save_quality_metrics=True):
        """
        Complete modality-specific preprocessing pipeline
        Integrates all preprocessing steps for a single image
        """
        original_image = self.load_image(image_path)
        if original_image is None:
            return None
        
        original_copy = original_image.copy()
        
        # Execute preprocessing pipeline
        brain_extracted = self.extract_brain_region(original_image, modality)
        denoised = self.medical_denoise(brain_extracted, modality)
        enhanced = self.advanced_contrast_enhancement(denoised, modality)
        processed = self.multi_scale_processing(enhanced)
        processed = processed.astype(np.float32) / 255.0  # Normalization to [0, 1]
        
        if save_quality_metrics:
            image_name = Path(image_path).name
            self.calculate_quality_metrics(original_copy, 
                                         (processed[:,:,0] * 255).astype(np.uint8), 
                                         image_name)
        
        return processed
    
    def apply_data_augmentation(self, image_path, modality, label, num_augmentations=5):
        """
        Apply data augmentation techniques for class balancing
        Includes random rotations and brightness/contrast adjustments
        """
        original_image = self.load_image(image_path)
        if original_image is None:
            return []
        
        augmented_images = []
        
        for i in range(num_augmentations):
            # Apply random augmentations
            if random.random() > 0.5:
                # Rotation (-15 to +15 degrees)
                angle = random.uniform(-15, 15)
                height, width = original_image.shape
                matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
                augmented = cv2.warpAffine(original_image, matrix, (width, height))
            else:
                # Brightness/contrast adjustment
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-10, 10)    # Brightness
                augmented = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)
            
            augmented_images.append(augmented)
        
        return augmented_images
    
    # ============================================================================
    # QUALITY ASSESSMENT METRICS
    # ============================================================================
    
    def calculate_quality_metrics(self, original, processed, image_name):
        """
        Calculate PSNR and SSIM metrics for quality assurance
        Tracks enhancement quality while preventing diagnostic value compromise
        """
        if original.shape != processed.shape:
            processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
        else:
            processed_resized = processed
            
        if original.dtype != processed_resized.dtype:
            processed_resized = processed_resized.astype(original.dtype)
            
        psnr = self.calculate_psnr(original, processed_resized)
        ssim = self.calculate_ssim(original, processed_resized)
        
        self.quality_metrics[image_name] = {
            'psnr': psnr,
            'ssim': ssim
        }
        
        return psnr, ssim
    
    def calculate_psnr(self, original, processed):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        Formula: PSNR = 20·log₁₀(255/√MSE)
        """
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_ssim(self, original, processed):
        """
        Simplified Structural Similarity Index (SSIM) calculation
        Evaluates luminance, contrast, and structure preservation
        """
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
        original = original.astype(float)
        processed = processed.astype(float)
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu_x = np.mean(original)
        mu_y = np.mean(processed)
        
        sigma_x = np.var(original)
        sigma_y = np.var(processed)
        sigma_xy = np.cov(original.flatten(), processed.flatten())[0, 1]
        
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        return numerator / denominator
    
    # ============================================================================
    # CLASS BALANCING FUNCTIONS
    # ============================================================================
    
    def detect_modality_and_label(self, path):
        """Detect modality (MRI/CT) and label (normal/tumor) from path"""
        path_str = str(path).lower()
        
        if "/mri/" in path_str:
            modality = "MRI"
        elif "/ct/" in path_str:
            modality = "CT"
        else:
            modality = "UNKNOWN"
        
        if "/normal/" in path_str:
            label = "normal"
        elif "/tumor/" in path_str:
            label = "tumor"
        else:
            label = None
        
        return modality, label
    
    def balance_dataset_strategy(self, all_images):
        """
        Apply strategic balancing to handle class imbalance
        Augments normal class to match tumor class distribution
        """
        print("Applying class balancing")
        
        # Count images by label
        label_counts = Counter(label for _, _, label in all_images)
        print(f" Original distribution: {dict(label_counts)}")
        
        normal_count = label_counts['normal']
        tumor_count = label_counts['tumor']
        
        if normal_count == 0:
            print("No normal images found for balancing")
            return all_images
        
        # Strategy: Augment normal class to match tumor class
        augmentation_factor = tumor_count // normal_count
        print(f" Augmentation factor: {augmentation_factor}")
        
        balanced_data = []
        
        # Keep all original data
        balanced_data.extend(all_images)
        
        # Augment normal class
        normal_images = [(path, mod, label) for path, mod, label in all_images if label == 'normal']
        
        print(f" Augmenting normal class...")
        for img_path, modality, label in tqdm(normal_images, desc="Augmenting normal images"):
            # Apply multiple augmentations to each normal image
            augmented_images = self.apply_data_augmentation(img_path, modality, label, 
                                                          num_augmentations=augmentation_factor-1)
            
            for aug_img in augmented_images:
                # Create synthetic file path for augmented image
                synthetic_path = f"synthetic_{Path(img_path).stem}_{len(balanced_data)}.png"
                balanced_data.append((synthetic_path, modality, label))
        
        # Count balanced distribution
        balanced_counts = Counter(label for _, _, label in balanced_data)
        print(f" Balanced distribution: {dict(balanced_counts)}")
        
        return balanced_data
    
    # ============================================================================
    # MAIN PROCESSING PIPELINE
    # ============================================================================
    
    def execute_complete_pipeline(self, input_base_dir, output_base_dir):
        """
        Execute complete multimodal preprocessing pipeline:
        1. Dataset reorganization and standardization
        2. Modality-specific preprocessing
        3. Class balancing
        4. Train/val/test splitting
        5. Quality assessment
        """
        
        print("STARTING COMPLETE MULTIMODAL PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Dataset Reorganization
        print("\n PHASE 1: DATASET REORGANIZATION & STANDARDIZATION")
        print("-" * 40)
        
        # First, examine current structure
        self.list_directory_structure(input_base_dir)
        
        # Reorganize folders
        file_counts = self.organize_mri_ct_folders(input_base_dir)
        
        # Step 2: Collect and prepare images for preprocessing
        print("\n PHASE 2: IMAGE COLLECTION & PREPARATION")
        print("-" * 40)
        
        # Create output directories for processed data
        for modality in ["MRI", "CT"]:
            for split in ["train", "val", "test"]:
                for label in ["tumor", "normal"]:
                    os.makedirs(f"{output_base_dir}/{modality}/{split}/{label}", exist_ok=True)
        
        # Supported extensions
        valid_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"]
        
        # Collect all reorganized images
        print("Scanning multimodal images...")
        all_images = []
        
        for modality in ["MRI", "CT"]:
            modality_path = Path(input_base_dir) / modality
            if modality_path.exists():
                for label in ["normal", "tumor"]:
                    label_path = modality_path / label
                    if label_path.exists():
                        images = [
                            (p, modality, label) for p in label_path.rglob("*")
                            if p.is_file() and p.suffix.lower() in valid_ext
                        ]
                        all_images.extend(images)
                        print(f" Found {len(images)} {modality} {label} images")
        
        print(f"Total usable images after reorganization: {len(all_images)}")
        
        if len(all_images) == 0:
            print("No images found after reorganization!")
            return
        
        # Step 3: Class Balancing (if enabled)
        print("\n PHASE 3: CLASS BALANCING")
        print("-" * 40)
        
        if self.apply_balancing:
            balanced_data = self.balance_dataset_strategy(all_images)
        else:
            balanced_data = all_images
        
        # Separate by modality for balanced splitting
        mri_data = [(path, label) for path, mod, label in balanced_data if mod == "MRI" and isinstance(path, Path)]
        ct_data = [(path, label) for path, mod, label in balanced_data if mod == "CT" and isinstance(path, Path)]
        
        # For synthetic images
        synthetic_data = [(path, mod, label) for path, mod, label in balanced_data if isinstance(path, str) and path.startswith("synthetic_")]
        
        print(f" Real MRI images: {len(mri_data)}")
        print(f" Real CT images: {len(ct_data)}")
        print(f" Synthetic images: {len(synthetic_data)}")
        
        # Step 4: Dataset Splitting
        print("\n PHASE 4: DATASET SPLITTING (70%/15%/15%)")
        print("-" * 40)
        
        splits = {}
        
        for modality_data, modality_name in [(mri_data, "MRI"), (ct_data, "CT")]:
            if modality_data:
                # Shuffle and split
                random.shuffle(modality_data)
                train_split = int(0.7 * len(modality_data))
                val_split = int(0.85 * len(modality_data))
                
                splits[modality_name] = {
                    "train": modality_data[:train_split],
                    "val": modality_data[train_split:val_split],
                    "test": modality_data[val_split:]
                }
                
                print(f"{modality_name}: Train={len(splits[modality_name]['train'])}, "
                      f"Val={len(splits[modality_name]['val'])}, "
                      f"Test={len(splits[modality_name]['test'])}")
        
        # Step 5: Modality-Specific Preprocessing
        print("\n PHASE 5: MODALITY-SPECIFIC PREPROCESSING")
        print("-" * 40)
        
        for modality, modality_splits in splits.items():
            print(f"\n Processing {modality} images...")
            
            for split_name, split_data in modality_splits.items():
                print(f"   Processing {split_name} set ({len(split_data)} images)...")
                
                for img_path, label in tqdm(split_data, desc=f"Processing {modality} {split_name}"):
                    try:
                        processed = self.modality_specific_preprocessing(img_path, modality)
                        
                        if processed is not None:
                            # Save as .npy for model training
                            filename = f"{Path(img_path).stem}_processed.npy"
                            output_path = f"{output_base_dir}/{modality}/{split_name}/{label}/{filename}"
                            
                            np.save(output_path, processed)
                            
                            # Also save as PNG for visualization
                            image_output_path = output_path.replace('.npy', '.png')
                            cv2.imwrite(image_output_path, (processed[:,:,0] * 255).astype(np.uint8))
                            
                    except Exception as e:
                        print(f" Error processing {img_path}: {e}")
                        continue
        
        # Step 6: Generate Comprehensive Report
        print("\n PHASE 6: COMPREHENSIVE REPORT GENERATION")
        print("=" * 60)
        
        # Count final distribution
        final_counts = {'MRI': {'normal': 0, 'tumor': 0}, 'CT': {'normal': 0, 'tumor': 0}}
        for modality in ["MRI", "CT"]:
            for label in ["normal", "tumor"]:
                for split in ["train", "val", "test"]:
                    split_path = Path(output_base_dir) / modality / split / label
                    if split_path.exists():
                        count = len(list(split_path.glob("*.npy")))
                        final_counts[modality][label] += count
        
        print("\n FINAL DATASET DISTRIBUTION:")
        print("-" * 40)
        
        total_images = 0
        for modality, counts in final_counts.items():
            modality_total = counts['normal'] + counts['tumor']
            total_images += modality_total
            print(f"\n{modality}:")
            print(f"  Normal: {counts['normal']}")
            print(f"  Tumor: {counts['tumor']}")
            print(f"  Total: {modality_total}")
            if counts['normal'] > 0:
                print(f"  Tumor/Normal Ratio: {counts['tumor']/counts['normal']:.2f}:1")
        
        print(f"\n GRAND TOTAL IMAGES PROCESSED: {total_images}")
        
        # Quality Metrics Summary
        print("\n QUALITY METRICS SUMMARY:")
        print("-" * 40)
        if self.quality_metrics:
            psnr_values = [metrics['psnr'] for metrics in self.quality_metrics.values() if metrics['psnr'] != float('inf')]
            ssim_values = [metrics['ssim'] for metrics in self.quality_metrics.values()]
            
            if psnr_values:
                print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
                print(f"PSNR Range: {min(psnr_values):.2f} - {max(psnr_values):.2f} dB")
            if ssim_values:
                print(f"Average SSIM: {np.mean(ssim_values):.4f}")
                print(f"SSIM Range: {min(ssim_values):.4f} - {max(ssim_values):.4f}")
        
        print(f"\n COMPLETE PIPELINE EXECUTION FINISHED!")
        print(f" Processed datasets saved in: {output_base_dir}")
        
        return final_counts, self.quality_metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_BASE_DIR = "/Users/jeissonparra/Documents/Master_s Degree Florida International University/Data Science & AI/Spring - 2026/Capstone/Datasets/Original"
    OUTPUT_BASE_DIR = "/Users/jeissonparra/Documents/Master_s Degree Florida International University/Data Science & AI/Spring - 2026/Capstone/Datasets/Balanced_Multimodal"
    
    # Create pipeline instance
    pipeline = MultimodalBrainTumorPreprocessingPipeline(
        target_size=(224, 224),
        apply_balancing=True
    )
    
    # Execute complete pipeline
    try:
        final_counts, quality_metrics = pipeline.execute_complete_pipeline(
            input_base_dir=INPUT_BASE_DIR,
            output_base_dir=OUTPUT_BASE_DIR
        )
        print("\n All preprocessing steps completed successfully!")
    except Exception as e:
        print(f"\n Pipeline execution failed with error: {e}")
        import traceback
        traceback.print_exc()