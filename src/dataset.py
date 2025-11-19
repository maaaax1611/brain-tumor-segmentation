import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib

class BraTSDataset(Dataset):
    """
    Custom Dataset for BraTS 2019 (HGG + LGG)
    """
    def __init__(self, root_dir, slice_axis=2, phase='train'):
        """
        Args:
            root_dir (string): Path to 'MICCAI_BraTS_2019_Data_Training'.
                               It must contain 'HGG' and 'LGG' subfolders.
            slice_axis (int): 2 = Axial (standard view).
            phase (string): 'train' or 'val'.
        """
        self.root_dir = root_dir
        self.slice_axis = slice_axis
        self.phase = phase
        self.slices_per_volume = 155

        # 1. Collect all patient paths from both HGG and LGG folders
        self.patient_paths = []
        
        # Check HGG folder
        hgg_path = os.path.join(root_dir, 'HGG')
        if os.path.exists(hgg_path):
            for p_id in os.listdir(hgg_path):
                full_path = os.path.join(hgg_path, p_id)
                if os.path.isdir(full_path):
                    self.patient_paths.append(full_path)

        # Check LGG folder
        lgg_path = os.path.join(root_dir, 'LGG')
        if os.path.exists(lgg_path):
            for p_id in os.listdir(lgg_path):
                full_path = os.path.join(lgg_path, p_id)
                if os.path.isdir(full_path):
                    self.patient_paths.append(full_path)
        
        print(f"Dataset loaded: Found {len(self.patient_paths)} patients in HGG+LGG.")

    def __len__(self):
        return len(self.patient_paths) * self.slices_per_volume

    def __getitem__(self, idx):
        patient_idx = idx // self.slices_per_volume
        slice_idx = idx % self.slices_per_volume
        
        patient_path = self.patient_paths[patient_idx]
        
        # ROBUST LOADING: Use glob to find files ending with specific suffixes
        # This handles naming variations and path issues automatically.
        try:
            flair_path = glob.glob(os.path.join(patient_path, '*flair.nii'))[0]
            t1ce_path = glob.glob(os.path.join(patient_path, '*t1ce.nii'))[0]
            seg_path = glob.glob(os.path.join(patient_path, '*seg.nii'))[0]
            
            flair = nib.load(flair_path).get_fdata()
            t1ce = nib.load(t1ce_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()
        except Exception as e:
            print(f"Error loading patient at {patient_path}: {e}")
            # Return zeros to keep the dataloader from crashing, but log the error
            return torch.zeros(2, 240, 240), torch.zeros(1, 240, 240)

        # Extract Slice
        if self.slice_axis == 2:
            img_flair = flair[:, :, slice_idx]
            img_t1ce = t1ce[:, :, slice_idx]
            mask = seg[:, :, slice_idx]

        # Normalize (Standardize)
        img_flair = self.normalize(img_flair)
        img_t1ce = self.normalize(img_t1ce)

        # Stack (2 channels: Flair + T1ce)
        image = np.stack([img_flair, img_t1ce], axis=0)

        # Process Mask: Combine labels 1, 2, 4 into "Tumor" (1)
        mask[mask > 0] = 1
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

    def normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

if __name__ == "__main__":
    # Test the loader (Adjust path to where you unzipped it!)
    # e.g., "D:/projects/brain-tumor-segmentation/data/MICCAI_BraTS_2019_Data_Training"
    path = "./data/MICCAI_BraTS_2019_Data_Training" 
    
    if os.path.exists(path):
        ds = BraTSDataset(path)
        img, mask = ds[100] # Get slice 100 of first patient
        print(f"Image Shape: {img.shape}") # Should be (2, 240, 240)
        print(f"Mask Shape: {mask.shape}")  # Should be (1, 240, 240)
    else:
        print("Path not found. Please adjust variable 'path'.")