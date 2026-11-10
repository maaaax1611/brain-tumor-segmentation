"""Patient-level train/val/test split.

Splitting by slice leaks information: neighbouring slices of the same patient are
nearly identical, so a slice-level random split puts "the same" data into train
and validation and inflates the score. We therefore split by patient and store
the split in a JSON file so every script (train, evaluate, predict) uses the
same, reproducible partition.
"""
import os
import json
import random
import argparse

from src.config import PROCESSED_DIR, SPLIT_FILE
from src.dataset import index_slices


def make_split(patient_ids, val_frac=0.1, test_frac=0.1, seed=42):
    ids = sorted(patient_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
    if n_test + n_val >= n:
        raise ValueError(f"Not enough patients ({n}) for the requested split.")
    return {
        "seed": seed,
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test:n_test + n_val]),
        "train": sorted(ids[n_test + n_val:]),
    }


def save_split(split, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path):
    with open(path) as f:
        return json.load(f)


def get_or_create_split(processed_dir, path, val_frac=0.1, test_frac=0.1, seed=42):
    if os.path.exists(path):
        split = load_split(path)
        print(f"Using existing split {path}: "
              f"{len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test patients")
        return split
    patients = list(index_slices(processed_dir))
    if not patients:
        raise RuntimeError(f"No preprocessed slices found in {processed_dir}")
    split = make_split(patients, val_frac, test_frac, seed)
    save_split(split, path)
    print(f"Created new split {path}: "
          f"{len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test patients")
    return split


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create a patient-level data split")
    p.add_argument("--data_path", default=PROCESSED_DIR)
    p.add_argument("--split_file", default=SPLIT_FILE)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing split file")
    a = p.parse_args()
    if a.force and os.path.exists(a.split_file):
        os.remove(a.split_file)
    get_or_create_split(a.data_path, a.split_file, a.val_split, a.test_split, a.seed)
