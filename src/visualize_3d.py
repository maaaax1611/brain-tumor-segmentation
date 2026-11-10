import os
import argparse
import torch
import numpy as np
import pyvista as pv
import glob
from tqdm import tqdm
from skimage import measure

from src.config import N_CHANNELS, N_CLASSES, PROCESSED_DIR, SPLIT_FILE, CHECKPOINT_DIR
from src.dataset import load_patient_volume
from src.metrics import dice_3d, predict_volume as predict_volume_batched
from src.models import UNet
from src.splits import load_split

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = PROCESSED_DIR
CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

def load_volume(patient_id):
    """Stack all slices of a patient -> (D, C, H, W) image and (D, 1, H, W) mask."""
    print(f"Reconstructing 3D volume for Patient: {patient_id}...")
    return load_patient_volume(DATA_DIR, patient_id)


def predict_volume(model, vol_numpy):
    """Slice-wise inference over the whole volume -> binary (D, H, W)."""
    print("Running Inference on 3D Volume...")
    return predict_volume_batched(model, vol_numpy, DEVICE)


def clean_prediction(pred_vol):
    """
    Finds the largest connected tumor cluster and removes smaller artifacts (noise).
    This ensures the predicted tumor is one single, solid object for visualization.
    """
    # Label all islands/regions in the 3D volume
    labels = measure.label(pred_vol)
    
    if labels.max() == 0: 
        return pred_vol # Return volume if no tumor is found
        
    # Measure properties of all regions
    regions = measure.regionprops(labels)
    
    # Find the region with the largest area (the true tumor)
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create a new mask containing only the largest component
    cleaned_mask = (labels == largest_region.label).astype(np.float32)
    
    return cleaned_mask


def visualize(vol, mask, pred, patient_id):
    """
    Create the 3D Visualization: Surface Mesh (Glass Brain) + Tumor.
    Calculate the threshold based on tissue density and use 
    clipping to remove the external bounding box artifacts (planes/cube).
    """
    print("Rendering 3D Scene...")
    
    # clean prediction
    pred_clean = clean_prediction(pred)

    # FLAIR-Channel (Channel 0) contains brain structure
    brain_vol = vol[:, 0, :, :] 
    grid = pv.wrap(brain_vol)

    # --- 1. EXTRACTING THE ORGANIC TISSUE (Ignoring the Skull/Padding) ---
    
    # Filter out absolute background (Min + epsilon) to find true tissue pixels
    valid_pixels = brain_vol[brain_vol > (brain_vol.min() + 1e-6)] 
    
    if valid_pixels.size == 0:
        print("ERROR: Could not find valid brain tissue (pixels > Min). Visualization aborted.")
        return 

    # Calculate threshold based on 70% of the mean tissue intensity 
    # This ignores low-density padding and focuses on dense tissue.
    mean_brain_intensity = valid_pixels.mean() 
    threshold_value = mean_brain_intensity * 0.75
    
    print(f"Calculated Brain Surface Threshold: {threshold_value:.4f}")

    # Create Initial Mesh
    brain_mesh = grid.threshold(threshold_value)
    brain_surf = brain_mesh.extract_surface()
    brain_surf = brain_surf.smooth(n_iter=50) # Smooth the surface for an organic look

    # --- 2. CLIP ARTIFACTS (Remove outer planes/cube artifacts) ---
    
    # Determine the actual bounds of the newly created mesh
    x_min, x_max, y_min, y_max, z_min, z_max = brain_surf.bounds
    
    # Define a small margin (5%) to cut the flat edges created by the thresholding
    clip_x_margin = (x_max - x_min) * 0.12
    clip_y_margin = (y_max - y_min) * 0.11
    clip_z_margin = (z_max - z_min) * 0.11

    # Define the new, slightly smaller bounds box
    clipped_bounds = [
        x_min + clip_x_margin, x_max - clip_x_margin,
        y_min + clip_y_margin, y_max - clip_y_margin,
        z_min + clip_z_margin, z_max - clip_z_margin
    ]

    # Clip the mesh using the calculated box
    brain_surf = brain_surf.clip_box(clipped_bounds, invert=False)
    
    # --- 3. RENDERING ---
    p = pv.Plotter(shape=(1, 2)) 
    p.set_background("white") 

    # --- Fenster 1: Ground Truth ---
    p.subplot(0, 0)
    p.add_text("Ground Truth", font_size=12)
    
    # 1. Brain
    p.add_mesh(brain_surf, 
               color="lightgray", 
               opacity=0.1,          # Very slight transparency
               style='surface',      
               smooth_shading=True)
    
    # 2. Tumor (Green)
    grid_mask = pv.wrap(mask[:, 0, :, :]) 
    p.add_mesh(grid_mask.threshold(0.5), color="#00ff00", opacity=1.0, label="Tumor GT")

    # --- Fenster 2: Prediction ---
    p.subplot(0, 1)
    p.add_text(f"Prediction", font_size=12)
    
    # 1. Brain
    p.add_mesh(brain_surf, 
               color="lightgray", 
               opacity=0.1, 
               style='surface',
               smooth_shading=True)
    
    # 2. Tumor (Red)
    grid_pred = pv.wrap(pred_clean)
    p.add_mesh(grid_pred.threshold(0.5), color="#ff0000", opacity=1.0, label="AI Prediction")
    
    p.link_views()
    
    # # --- GIF EXPORT ---
    # # Create an output directory if it doesn't exist
    # OUTPUT_DIR = "visualizations"
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # gif_path = os.path.join(OUTPUT_DIR, f"{patient_id}_3d_segmentation.gif")

    # print(f"Exporting GIF animation to: {gif_path}")
    
    # # Define a simple camera path
    # # Startpunkt der Kamera: leicht von der Seite
    # p.camera_position = [(400, 200, 200), (120, 120, 75), (0, 1, 0)] 
    
    # # Eine Rotation um 360 Grad um die Y-Achse
    # # 30 Frames für eine flüssige Animation
    # # loop=True sorgt dafür, dass es unendlich läuft
    # p.open_gif(gif_path, fps=15) # 15 Frames pro Sekunde
    
    # # 360 Grad in 30 Schritten = 12 Grad pro Schritt
    # n_frames = 30
    # for i in range(n_frames):
    #     p.camera.azimuth += 360 / n_frames # Drehe die Kamera um 360 Grad
    #     p.write_frame() # Schreibe den aktuellen Frame ins GIF

    # p.close_gif()
    # print(f"GIF export completed for patient {patient_id}.")
    # # --- END GIF EXPORT ---

    p.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default=None, help="Patient ID (default: first test patient)")
    args = parser.parse_args()

    # 1. Load Model
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    # 2. Pick a held-out test patient so the visualization is not on training data
    patient_id = args.patient or load_split(SPLIT_FILE)["test"][0]

    # 3. Run Pipeline
    vol, mask = load_volume(patient_id)
    pred = predict_volume(model, vol)
    print(f"Whole-tumor Dice for {patient_id}: {dice_3d(pred, mask[:, 0]):.4f}")

    visualize(vol, mask, pred, patient_id)

if __name__ == "__main__":
    main()