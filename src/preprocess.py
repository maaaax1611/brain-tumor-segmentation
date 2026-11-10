import os
import glob
import argparse

import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.config import MODALITIES, RAW_DATA_DIR, PROCESSED_DIR


def normalize_volume(vol):
    """Z-score normalization over the brain region of the whole 3D volume.

    Statistics are computed on non-zero voxels (the skull-stripped brain) so the
    black background does not bias mean/std. Background stays at 0. Using
    per-volume instead of per-slice statistics keeps intensities consistent
    across slices of the same patient.
    """
    brain = vol > 0
    if brain.sum() == 0:
        return vol.astype(np.float32)
    mean = vol[brain].mean()
    std = vol[brain].std()
    out = np.zeros_like(vol, dtype=np.float32)
    out[brain] = (vol[brain] - mean) / (std + 1e-8)
    return out


def find_modality(patient_path, modality):
    # Use the "_<mod>.nii" suffix so that "t1" does not match "t1ce".
    matches = glob.glob(os.path.join(patient_path, f"*_{modality}.nii*"))
    if not matches:
        raise FileNotFoundError(f"No '{modality}' file in {patient_path}")
    return matches[0]


def preprocess_dataset(root_dir, output_dir):
    print(f"Preprocessing data from {root_dir} to {output_dir}")
    print(f"Modalities (channel order): {MODALITIES}")
    os.makedirs(output_dir, exist_ok=True)

    patient_paths = []
    for subfolder in ["HGG", "LGG"]:
        path = os.path.join(root_dir, subfolder)
        if os.path.exists(path):
            patient_paths.extend(sorted(glob.glob(os.path.join(path, "*"))))

    print(f"Found {len(patient_paths)} patients. Starting conversion...")
    total_slices = 0

    for patient_path in tqdm(patient_paths):
        patient_id = os.path.basename(patient_path)
        try:
            vols = [normalize_volume(nib.load(find_modality(patient_path, m)).get_fdata())
                    for m in MODALITIES]
            seg_vol = nib.load(find_modality(patient_path, "seg")).get_fdata()

            image_vol = np.stack(vols, axis=0).astype(np.float32)  # (C, H, W, D)
            # Keep the original BraTS label map (0, 1, 2, 4); the dataset turns
            # it into region channels, so the target definition can change
            # without re-running preprocessing.
            mask_vol = np.rint(seg_vol).astype(np.uint8)

            for i in range(image_vol.shape[-1]):
                save_name = f"{patient_id}_{i:03d}"
                np.save(os.path.join(output_dir, f"{save_name}_img.npy"), image_vol[..., i])
                np.save(os.path.join(output_dir, f"{save_name}_mask.npy"), mask_vol[None, :, :, i])
                total_slices += 1
        except Exception as e:
            print(f"Skipping {patient_id}: {e}")

    print(f"Done! Saved {total_slices} slices to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--output_path", type=str, default=PROCESSED_DIR)
    args = parser.parse_args()
    preprocess_dataset(args.data_path, args.output_path)
