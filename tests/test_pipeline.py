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

from src.config import MODALITIES, N_CHANNELS, N_CLASSES, REGION_NAMES
from src.dataset import (BraTSDataset, index_slices, labels_to_regions, load_patient_volume,
                         parse_slice_filename, regions_to_labelmap)
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
                vol[8:20, 8:20, 1:6] = 2    # edema
                vol[10:18, 10:18, 2:5] = 1  # necrotic / non-enhancing core
                vol[12:16, 12:16, 3:4] = 4  # enhancing tumor
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


def test_regions():
    labels = np.array([[0, 1, 2, 4]])
    regions = labels_to_regions(labels)                      # (R, 1, 4)
    assert regions.shape == (3, 1, 4)
    assert regions[0].tolist() == [[0, 1, 1, 1]]             # WT
    assert regions[1].tolist() == [[0, 1, 0, 1]]             # TC
    assert regions[2].tolist() == [[0, 0, 0, 1]]             # ET
    assert regions_to_labelmap(regions).tolist() == [[0, 2, 1, 3]]
    # nesting is enforced: ET outside WT is dropped
    bad = np.zeros((3, 1, 2)); bad[2, 0, 0] = 1
    assert regions_to_labelmap(bad).sum() == 0


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
        assert img.shape == (N_CHANNELS, H, W) and mask.shape == (N_CLASSES, H, W)
        assert set(np.unique(mask.numpy())) <= {0.0, 1.0}
        assert {parse_slice_filename(f)[0] for f in ds.image_files} == set(split["train"])

        vol, m = load_patient_volume(proc, split["test"][0])
        assert vol.shape == (D, N_CHANNELS, H, W) and m.shape == (D, N_CLASSES, H, W)
        assert m[:, 0].sum() > m[:, 1].sum() > m[:, 2].sum() > 0   # WT ⊃ TC ⊃ ET
        # per-volume normalization: brain region has ~zero mean / unit std per channel
        brain = vol[:, 0] != 0
        assert abs(vol[:, 0][brain].mean()) < 1e-3 and abs(vol[:, 0][brain].std() - 1) < 1e-2

        model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
        logits = model(img.unsqueeze(0))
        assert logits.shape == (1, N_CLASSES, H, W)
        scores = evaluate_patients(model, proc, split["val"], "cpu", progress=False)
        assert set(scores) == set(split["val"])
        assert all(set(s) == set(REGION_NAMES) and all(0 <= v <= 1 for v in s.values()) for s in scores.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
