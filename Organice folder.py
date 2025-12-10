import os
import shutil
from pathlib import Path

def organize_mri_ct_folders(base_dir):
    """
    Reorganize MRI and CT folders into a standardized structure:
    - MRI/normal
    - MRI/tumor
    - CT/normal  
    - CT/tumor
    
    Preserve tumor type information in filenames for glioma, meningioma, and pituitary tumors.
    """
    
    print(f"Working in base directory: {base_dir}")
    print(f"Base directory exists: {os.path.exists(base_dir)}")
    
    # Define tumor type mappings
    tumor_type_mapping = {
        'meningioma_tumor': '1',
        'glioma_tumor': '2', 
        'pituitary_tumor': '3'
    }
    
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
        """Process individual folders and copy/rename files"""
        
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
                
                # Create new filename
                file_ext = os.path.splitext(filename)[1]
                
                if is_tumor and tumor_type and tumor_type in tumor_type_mapping:
                    # Include tumor type code in filename
                    new_filename = f"{category}_{file_counters[category]:04d}_{tumor_type_mapping[tumor_type]}{file_ext}"
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
    print("Processing CT folders...")
    
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

def list_directory_structure(base_dir, max_depth=3):
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

# Usage
if __name__ == "__main__":
    # Use the correct path to your dataset
    base_directory = "/Users/jeissonparra/Library/CloudStorage/OneDrive-FloridaInternationalUniversity/Capstone/Multimodal_Deep Learning_for_Early_Detection_and_Classification_of_Brain_Tumors_Using_MRI_and_CT_Scans/Datasets"
    
    # First, let's see what's in the directory
    list_directory_structure(base_directory)
    
    print("\n" + "="*50)
    print("Starting folder reorganization...")
    organize_mri_ct_folders(base_directory)
    
    print("Done!")