"""Evaluation metrics, computed per patient on the full 3D volume (as in BraTS)."""
import numpy as np
import torch
from tqdm import tqdm

from src.config import REGION_NAMES
from src.dataset import index_slices, load_patient_volume


def dice_3d(pred, target):
    """Dice of two binary volumes.

    Convention for the empty case (no tumor in the ground truth): 1.0 if the
    prediction is also empty, else 0.0. No smoothing term, so empty slices or
    volumes cannot inflate the score.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / denom)


@torch.no_grad()
def predict_volume(model, vol, device, batch_size=32, threshold=0.5):
    """Slice-wise inference. vol: (D, C, H, W) numpy -> binary region masks (D, R, H, W)."""
    model.eval()
    preds = []
    for i in range(0, len(vol), batch_size):
        x = torch.from_numpy(vol[i:i + batch_size]).float().to(device)
        probs = torch.sigmoid(model(x))
        preds.append((probs > threshold).cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def evaluate_patients(model, processed_dir, patient_ids, device, batch_size=32, progress=True):
    """Returns {patient_id: {"WT": dice, "TC": dice, "ET": dice}}."""
    index = index_slices(processed_dir)
    results = {}
    it = tqdm(patient_ids, desc="Eval (3D)") if progress else patient_ids
    for pid in it:
        vol, mask = load_patient_volume(processed_dir, pid, index)
        pred = predict_volume(model, vol, device, batch_size)
        results[pid] = {name: dice_3d(pred[:, r], mask[:, r]) for r, name in enumerate(REGION_NAMES)}
    return results


def mean_per_region(results):
    """{pid: {region: dice}} -> {region: mean dice over patients}."""
    return {name: float(np.mean([r[name] for r in results.values()])) for name in REGION_NAMES}
