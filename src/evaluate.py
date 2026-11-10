"""Evaluate a checkpoint on the held-out patients (default: test split).

Reports whole-tumor Dice per patient on the full 3D volume and writes a CSV.

    python -m src.evaluate --checkpoint checkpoints/best_model.pth --subset test
"""
import os
import csv
import argparse

import numpy as np
import torch

from src.config import N_CHANNELS, N_CLASSES, PROCESSED_DIR, SPLIT_FILE, CHECKPOINT_DIR
from src.metrics import evaluate_patients
from src.models import UNet
from src.splits import load_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    p.add_argument("--data_path", default=PROCESSED_DIR)
    p.add_argument("--split_file", default=SPLIT_FILE)
    p.add_argument("--subset", default="test", choices=["train", "val", "test"])
    p.add_argument("--out_csv", default=None, help="Default: results_<subset>.csv next to the checkpoint")
    a = p.parse_args()

    patients = load_split(a.split_file)[a.subset]
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(a.checkpoint, map_location=DEVICE))

    scores = evaluate_patients(model, a.data_path, patients, DEVICE)
    vals = np.array(list(scores.values()))

    print("-" * 40)
    print(f"Whole-tumor Dice on {a.subset} ({len(vals)} patients)")
    print(f"  mean   {vals.mean():.4f} ± {vals.std():.4f}")
    print(f"  median {np.median(vals):.4f}")
    print(f"  min    {vals.min():.4f}  ({min(scores, key=scores.get)})")
    print("-" * 40)

    out_csv = a.out_csv or os.path.join(os.path.dirname(a.checkpoint) or ".", f"results_{a.subset}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "dice_wt"])
        for pid, d in sorted(scores.items()):
            w.writerow([pid, f"{d:.4f}"])
    print(f"Per-patient results written to {out_csv}")


if __name__ == "__main__":
    main()
