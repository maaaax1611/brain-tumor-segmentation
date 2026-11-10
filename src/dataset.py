import os
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_slice_filename(path):
    """'BraTS19_2013_10_1_075_img.npy' -> ('BraTS19_2013_10_1', 75)."""
    stem = os.path.basename(path)[: -len("_img.npy")]
    patient_id, slice_idx = stem.rsplit("_", 1)
    return patient_id, int(slice_idx)


def index_slices(processed_dir):
    """Map patient_id -> list of image file paths, sorted by slice index."""
    index = defaultdict(list)
    for f in glob.glob(os.path.join(processed_dir, "*_img.npy")):
        pid, idx = parse_slice_filename(f)
        index[pid].append((idx, f))
    return {pid: [f for _, f in sorted(items)] for pid, items in index.items()}


class BraTSDataset(Dataset):
    """2D slice dataset over preprocessed .npy files.

    Args:
        processed_dir: folder with *_img.npy / *_mask.npy files.
        patient_ids: optional list of patients to include. Use this to build
            train/val/test sets so that slices of one patient never end up in
            more than one split.
    """

    def __init__(self, processed_dir, patient_ids=None):
        self.processed_dir = processed_dir
        index = index_slices(processed_dir)
        if patient_ids is not None:
            missing = set(patient_ids) - set(index)
            if missing:
                print(f"Warning: {len(missing)} patients from the split have no slices.")
            index = {pid: index[pid] for pid in patient_ids if pid in index}

        self.patient_ids = sorted(index)
        self.image_files = [f for pid in self.patient_ids for f in index[pid]]
        print(f"Dataset: {len(self.patient_ids)} patients, {len(self.image_files)} slices.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = np.load(img_path)                                   # (C, H, W)
        mask = np.load(img_path.replace("_img.npy", "_mask.npy"))   # (1, H, W)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


def load_patient_volume(processed_dir, patient_id, index=None):
    """Stack all slices of one patient -> image (D, C, H, W), mask (D, 1, H, W)."""
    if index is None:
        index = index_slices(processed_dir)
    files = index.get(patient_id)
    if not files:
        raise ValueError(f"No slices found for patient {patient_id}")
    imgs = np.stack([np.load(f) for f in files])
    masks = np.stack([np.load(f.replace("_img.npy", "_mask.npy")) for f in files])
    return imgs, masks
