import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse

def normalize(slice_arr):
    """Standard Z-score normalization"""
    # Filter out background pixels for mean calculation to ignore black area
    mask = slice_arr > 0
    if mask.sum() == 0:
        return slice_arr
    mean = slice_arr[mask].mean()
    std = slice_arr[mask].std()
    return (slice_arr - mean) / (std + 1e-8)

def preprocess_dataset(root_dir, output_dir):
    print(f"Preprocessing data from {root_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect patient paths
    patient_paths = []
    for subfolder in ['HGG', 'LGG']:
        path = os.path.join(root_dir, subfolder)
        if os.path.exists(path):
            patient_paths.extend(glob.glob(os.path.join(path, "*")))
            
    print(f"Found {len(patient_paths)} patients. Starting conversion...")
    
    total_slices = 0
    
    for patient_path in tqdm(patient_paths):
        patient_id = os.path.basename(patient_path)
        
        try:
            # Load 3D Volumes
            flair_path = glob.glob(os.path.join(patient_path, '*flair.nii*'))[0]
            t1ce_path = glob.glob(os.path.join(patient_path, '*t1ce.nii*'))[0]
            seg_path = glob.glob(os.path.join(patient_path, '*seg.nii*'))[0]
            
            flair_vol = nib.load(flair_path).get_fdata()
            t1ce_vol = nib.load(t1ce_path).get_fdata()
            seg_vol = nib.load(seg_path).get_fdata()
            
            # Iterate over slices (Z-axis)
            # Standard BraTS Size is usually (240, 240, 155)
            n_slices = flair_vol.shape[2]
            
            for i in range(n_slices):
                # Check if slice has any tumor (or brain)? 
                # OPTIONAL: We could skip empty slices here to speed up training even more!
                # For now, let's keep all slices to be safe.
                
                # Extract 2D
                img_flair = flair_vol[:, :, i]
                img_t1ce = t1ce_vol[:, :, i]
                mask = seg_vol[:, :, i]
                
                # Normalize immediately
                img_flair = normalize(img_flair)
                img_t1ce = normalize(img_t1ce)
                
                # Stack Image (2, H, W)
                image_stacked = np.stack([img_flair, img_t1ce], axis=0).astype(np.float32)
                
                # Prepare Mask (1, H, W) -> Binary Tumor
                mask[mask > 0] = 1
                mask = np.expand_dims(mask, axis=0).astype(np.float32)
                
                # Save as .npy
                # Naming: PatientID_SliceIndex.npy
                save_name = f"{patient_id}_{i:03d}"
                
                np.save(os.path.join(output_dir, f"{save_name}_img.npy"), image_stacked)
                np.save(os.path.join(output_dir, f"{save_name}_mask.npy"), mask)
                
                total_slices += 1
                
        except Exception as e:
            print(f"Skipping {patient_id}: {e}")

    print(f"Done! Saved {total_slices} slices to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join('data', 'MICCAI_BraTS_2019_Data_Training'))
    parser.add_argument("--output_path", type=str, default=os.path.join('data', 'processed_slices'))
    args = parser.parse_args()
    
    preprocess_dataset(args.data_path, args.output_path)