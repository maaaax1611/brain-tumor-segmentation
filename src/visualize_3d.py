"""3D visualization of the tumor sub-regions: ground truth vs. prediction.

Renders a transparent brain surface with the three tumor compartments as
smooth, colored surfaces (edema, non-enhancing / necrotic core, enhancing tumor).

    python -m src.visualize_3d                          # interactive window, first test patient
    python -m src.visualize_3d --patient <ID>
    python -m src.visualize_3d --gif assets/tumor.gif   # rotating GIF, no window needed
    python -m src.visualize_3d --html tumor.html        # interactive scene for the browser
    python -m src.visualize_3d --gt_only                # no checkpoint needed
"""
import os
import argparse

import numpy as np
import pyvista as pv
import torch
from skimage import measure

from src.config import N_CHANNELS, N_CLASSES, PROCESSED_DIR, SPLIT_FILE, CHECKPOINT_DIR, REGION_NAMES
from src.dataset import load_patient_volume, regions_to_labelmap
from src.metrics import dice_3d, predict_volume
from src.models import UNet
from src.splits import load_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# label id in the display label map -> (name, color)
COMPARTMENTS = {
    1: ("Edema", "#2ecc71"),
    2: ("Non-enhancing / necrotic core", "#3498db"),
    3: ("Enhancing tumor", "#e74c3c"),
}
# The compartments are nested (edema around core around enhancing tumor), so the
# outer ones are drawn translucent to keep the inner ones visible.
OPACITY = {1: 0.18, 2: 0.3, 3: 1.0}


def clean_prediction(pred_regions):
    """Keep only the largest connected whole-tumor component (removes speckle).

    pred_regions: (D, R, H, W) binary. The component is found on the WT channel
    and applied to all regions.
    """
    labels = measure.label(pred_regions[:, 0] > 0.5)
    if labels.max() == 0:
        return pred_regions
    largest = max(measure.regionprops(labels), key=lambda r: r.area).label
    return pred_regions * (labels == largest)[:, None]


def surface(mask, smooth_iter=30):
    """Binary (D, H, W) volume -> smoothed surface mesh, or None if empty."""
    if mask.sum() == 0:
        return None
    # Pad so surfaces touching the volume border are closed.
    grid = pv.wrap(np.pad(mask.astype(np.float32), 1))
    mesh = grid.contour([0.5])
    if mesh.n_points == 0:
        return None
    return mesh.smooth(n_iter=smooth_iter)


def add_scene(p, brain_surf, labelmap, title):
    p.add_text(title, font_size=11, color="black")
    if brain_surf is not None:
        p.add_mesh(brain_surf, color="lightgray", opacity=0.08, smooth_shading=True)
    for lid, (name, color) in COMPARTMENTS.items():
        # Outer hull of each compartment (label >= lid) instead of a hollow shell:
        # one clean surface per nesting level.
        mesh = surface(labelmap >= lid)
        if mesh is not None:
            p.add_mesh(mesh, color=color, opacity=OPACITY[lid], smooth_shading=True, label=name)


def build_plotter(vol, gt_regions, pred_regions=None, patient_id="", off_screen=False):
    """vol: (D, C, H, W); *_regions: (D, R, H, W). Returns a ready pv.Plotter."""
    brain_surf = surface((vol != 0).any(axis=1), smooth_iter=50)
    gt_labels = regions_to_labelmap(gt_regions)

    n_cols = 1 if pred_regions is None else 2
    p = pv.Plotter(shape=(1, n_cols), off_screen=off_screen, window_size=(700 * n_cols, 700))
    p.set_background("white")

    p.subplot(0, 0)
    add_scene(p, brain_surf, gt_labels, f"Ground Truth\n{patient_id}")
    p.add_legend(bcolor=None, face="circle", size=(0.32, 0.14), loc="lower left")

    if pred_regions is not None:
        pred_regions = clean_prediction(pred_regions)
        dice = ", ".join(f"{n} {dice_3d(pred_regions[:, r], gt_regions[:, r]):.2f}"
                          for r, n in enumerate(REGION_NAMES))
        p.subplot(0, 1)
        add_scene(p, brain_surf, regions_to_labelmap(pred_regions), f"Prediction\nDice: {dice}")
        p.link_views()

    p.camera_position = "xz"
    p.camera.elevation = 20
    p.camera.zoom(1.2)
    return p


def write_gif(p, path, n_frames=60, fps=15):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    p.open_gif(path, fps=fps)
    for _ in range(n_frames):
        p.camera.azimuth += 360 / n_frames
        p.write_frame()
    p.close()
    print(f"GIF written to {path}")


def main():
    parser = argparse.ArgumentParser(description="3D view of tumor sub-regions (GT vs. prediction)")
    parser.add_argument("--patient", default=None, help="Patient ID (default: first test patient)")
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--data_path", default=PROCESSED_DIR)
    parser.add_argument("--split_file", default=SPLIT_FILE)
    parser.add_argument("--gt_only", action="store_true", help="Show only the ground truth (no model needed)")
    parser.add_argument("--gif", default=None, help="Write a rotating GIF to this path instead of opening a window")
    parser.add_argument("--html", default=None, help="Export an interactive HTML scene (needs trame)")
    args = parser.parse_args()

    # Held-out test patient by default, so the picture is not from training data.
    patient_id = args.patient or load_split(args.split_file)["test"][0]
    print(f"Loading volume for patient {patient_id}...")
    vol, gt_regions = load_patient_volume(args.data_path, patient_id)

    pred_regions = None
    if not args.gt_only:
        model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        print("Running inference...")
        pred_regions = predict_volume(model, vol, DEVICE)

    off_screen = bool(args.gif or args.html)
    p = build_plotter(vol, gt_regions, pred_regions, patient_id, off_screen=off_screen)

    if args.html:
        try:
            p.trame.export_html(args.html)   # PyVista >= 0.47 (trame-pyvista)
        except AttributeError:
            p.export_html(args.html)         # older PyVista
        print(f"Interactive scene written to {args.html}")
    if args.gif:
        write_gif(p, args.gif)
    elif not args.html:
        p.show()


if __name__ == "__main__":
    main()
