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
    