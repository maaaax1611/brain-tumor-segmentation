"""End-to-end smoke test on tiny synthetic BraTS-like volumes (no real data needed).

    python -m pytest tests/        # or: python -m tests.test_pipeline
"""
import os
import sys
import tempfile

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODALITIES, N_CHANNELS
from src.dataset import BraTSDataset, index_slices, load_patient_volume, parse_slice_filename
from src.metrics import dice_3d, evaluate_patients
from src.models import UNet
from src.preprocess import preprocess_dataset
from src.splits import make_split, get_or_create_split

H = W = 32
D = 8


def _make_fake_brats(root, n_patients=10):
    rng = np.random.default_rng(0)
    for i in range(n_patients):
        grade = "HGG" if i % 2 == 0 else "LGG"
        pid = f"BraTS19_FAKE_{i}_1"
        pdir = os.path.join(root, grade, pid)
        os.makedirs(pdir)
        brain = np.zeros((H, W, D))
        brain[4:28, 4:28, :] = 1
        for m in MODALITIES + ["seg"]:
            if m == "seg":
                vol = np.zeros((H, W, D))
                vol[10:16, 10:16, 2:5] = rng.choice([1, 2, 4])
            else:
                vol = brain * rng.uniform(100, 1000, (H, W, D))
            nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)), os.path.join(pdir, f"{pid}_{m}.nii.gz"))


def test_parse_filename():
    assert parse_slice_filename("/x/BraTS19_2013_10_1_075_img.npy") == ("BraTS19_2013_10_1", 75)


def test_dice_3d():
    a = np.zeros((4, 4, 4)); a[1:3, 1:3, 1:3] = 1
    assert dice_3d(a, a) == 1.0
    assert dice_3d(a, np.zeros_like(a)) == 0.0
    assert dice_3d(np.zeros_like(a), np.zeros_like(a)) == 1.0


def test_split_is_disjoint():
    ids = [f"p{i}" for i in range(100)]
    s = make_split(ids, 0.1, 0.1, seed=1)
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(ids)
    assert make_split(ids, 0.1, 0.1, seed=1) == s  # reproducible


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        raw, proc = os.path.join(tmp, "raw"), os.path.join(tmp, "proc")
        _make_fake_brats(raw)
        preprocess_dataset(raw, proc)

        index = index_slices(proc)
        assert len(index) == 10 and all(len(v) == D for v in index.values())

        split = get_or_create_split(proc, os.path.join(tmp, "splits.json"), 0.2, 0.2)
        ds = BraTSDataset(proc, split["train"])
        img, mask = ds[0]
        assert img.shape == (N_CHANNELS, H, W) and mask.shape == (1, H, W)
        assert set(np.unique(mask.numpy())) <= {0.0, 1.0}
        assert {parse_slice_filename(f)[0] for f in ds.image_files} == set(split["train"])

        vol, m = load_patient_volume(proc, split["test"][0])
        assert vol.shape == (D, N_CHANNELS, H, W)
        # per-volume normalization: brain region has ~zero mean / unit std per channel
        brain = vol[:, 0] != 0
        assert abs(vol[:, 0][brain].mean()) < 1e-3 and abs(vol[:, 0][brain].std() - 1) < 1e-2

        model = UNet(n_channels=N_CHANNELS, n_classes=1)
        logits = model(img.unsqueeze(0))
        assert logits.shape == (1, 1, H, W)
        scores = evaluate_patients(model, proc, split["val"], "cpu", progress=False)
        assert set(scores) == set(split["val"]) and all(0 <= v <= 1 for v in scores.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
