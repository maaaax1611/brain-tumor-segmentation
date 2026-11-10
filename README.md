# 🧠 Brain Tumor Segmentation (BraTS 2019)

![3D Visualization](assets/brain_rotation.gif)
> *3D visualization of the segmented tumor (red) vs. ground truth (green) inside the brain volume.*

Multi-region brain tumor segmentation on multi-modal MRI with a 2D U-Net, trained
slice-wise and evaluated per patient on the full 3D volume.

The model predicts the three standard BraTS regions, one sigmoid channel each:

| Region | BraTS labels | Meaning |
|---|---|---|
| **WT** – whole tumor | 1, 2, 4 | everything abnormal, incl. edema |
| **TC** – tumor core | 1, 4 | necrotic / non-enhancing core + enhancing tumor |
| **ET** – enhancing tumor | 4 | contrast-enhancing part |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| Preprocess | `src/preprocess.py` | Loads FLAIR, T1ce, T1, T2 + segmentation, z-score normalizes **per volume** (brain voxels only), saves 2D slices as `.npy` (4 × 240 × 240) plus the original BraTS label map. |
| Split | `src/splits.py` | **Patient-level** train / val / test split (default 80/10/10, seed 42), stored in `data/splits.json`. Created automatically on first training run. |
| Train | `src/train.py` | U-Net with 3 output channels (WT / TC / ET), BCE + per-region Dice loss, Adam. Validation = 3D Dice per region over validation patients; best model by the mean of the three. Saves `best_model.pth` / `last_model.pth`. |
| Evaluate | `src/evaluate.py` | 3D Dice per region and test patient (mean, std, median, worst case) + CSV. |
| Inspect | `src/predict.py`, `src/visualize_3d.py` | 2D slice view with region overlay and confidence map / 3D render of the tumor compartments (GT vs. prediction) — both on held-out test patients. |

### Why patient-level split and 3D Dice?
Neighbouring slices of one patient are almost identical. A random split over
slices puts the same anatomy into train and validation and overestimates
performance. Dice averaged over 2D batches is also biased: empty slices score
≈ 1 through the smoothing term. Both are fixed here; scores are therefore
comparable to how BraTS reports Dice.

## Usage

```bash
pip install -r requirements.txt

# BraTS 2019 training data → data/MICCAI_BraTS_2019_Data_Training/{HGG,LGG}/...
python -m src.preprocess
python -m src.train --epochs 20 --batch_size 16
python -m src.evaluate --subset test
```

### 3D visualization

Transparent brain surface with edema (green), non-enhancing / necrotic core
(blue) and enhancing tumor (red); ground truth and prediction side by side with
linked cameras and per-region Dice.

```bash
python -m src.visualize_3d                         # interactive window, first test patient
python -m src.visualize_3d --patient <ID>
python -m src.visualize_3d --gif assets/tumor.gif  # rotating GIF, no window needed
python -m src.visualize_3d --html tumor.html       # interactive scene for the browser
python -m src.visualize_3d --gt_only               # ground truth only, no checkpoint needed
```

Tests (synthetic data, no download needed):

```bash
python -m pytest tests/
```

## Results

3D Dice on the held-out test patients:

| Model | Input | WT | TC | ET |
|---|---|---|---|---|
| 2D U-Net | FLAIR, T1ce, T1, T2 | _tbd_ | _tbd_ | _tbd_ |

## Roadmap
- [x] Multi-class BraTS regions (WT / TC / ET)
- [ ] HD95
- [ ] Augmentation, LR scheduler, mixed precision
- [ ] 3D / 2.5D U-Net
- [ ] Transformer backbone (Swin UNETR) vs. U-Net baseline
- [ ] nnU-Net reference
