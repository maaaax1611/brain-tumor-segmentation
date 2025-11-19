import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    """
    Fast Dataset loading preprocessed .npy files.
    """
    def __init__(self, processed_dir):
        """
        Args:
            processed_dir (string): Path to the folder containing .npy files
        """
        self.processed_dir = processed_dir
        
        # Find all image files (we assume matching mask files exist)
        # Pattern: PatientID_SliceID_img.npy
        self.image_files = glob.glob(os.path.join(processed_dir, "*_img.npy"))
        
        print(f"Dataset loaded: Found {len(self.image_files)} preprocessed slices.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Construct mask path: replace '_img.npy' with '_mask.npy'
        mask_path = img_path.replace('_img.npy', '_mask.npy')
        
        try:
            # Load NumPy arrays (extremely fast)
            image = np.load(img_path) # Shape: (2, 240, 240)
            mask = np.load(mask_path) # Shape: (1, 240, 240)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(2, 240, 240), torch.zeros(1, 240, 240)

        # Convert to Tensor
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()