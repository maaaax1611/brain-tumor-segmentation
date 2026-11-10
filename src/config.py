"""Shared constants for the whole pipeline."""
import os

# Order of the input channels. Changing this requires re-running preprocessing.
MODALITIES = ["flair", "t1ce", "t1", "t2"]
N_CHANNELS = len(MODALITIES)
N_CLASSES = 1  # binary whole-tumor segmentation (all BraTS labels > 0)

RAW_DATA_DIR = os.path.join("data", "MICCAI_BraTS_2019_Data_Training")
PROCESSED_DIR = os.path.join("data", "processed_slices")
SPLIT_FILE = os.path.join("data", "splits.json")
CHECKPOINT_DIR = "checkpoints"
