"""Shared constants for the whole pipeline."""
import os

# Order of the input channels. Changing this requires re-running preprocessing.
MODALITIES = ["flair", "t1ce", "t1", "t2"]
N_CHANNELS = len(MODALITIES)

# BraTS label map: 1 = necrotic / non-enhancing core, 2 = edema, 4 = enhancing tumor.
# The model predicts the three (nested, overlapping) BraTS evaluation regions,
# one sigmoid output channel each:
REGIONS = {
    "WT": (1, 2, 4),  # whole tumor
    "TC": (1, 4),     # tumor core
    "ET": (4,),       # enhancing tumor
}
REGION_NAMES = list(REGIONS)
N_CLASSES = len(REGIONS)

RAW_DATA_DIR = os.path.join("data", "MICCAI_BraTS_2019_Data_Training")
PROCESSED_DIR = os.path.join("data", "processed_slices")
SPLIT_FILE = os.path.join("data", "splits.json")
CHECKPOINT_DIR = "checkpoints"
