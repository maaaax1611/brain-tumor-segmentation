import os
import torch
import numpy as np
import pyvista as pv
import glob
from tqdm import tqdm
from skimage import measure

# Imports
from src.models import UNet

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join('data', 'processed_slices')
CHECKPOINT = os.path.join('checkpoints', 'best_model.pth')

def load_volume(patient_id):
    """
    Load all slices of a patient and stack them to form a 3d volume.
    """
    print(f"Reconstructing 3D volume for Patient: {patient_id}...")
    
    # search for all files starting with the patient_id
    # pattern: data/processed_slices/BraTS19_2013_10_1_075_img.npy
    pattern = os.path.join(DATA_DIR, f"{patient_id}_*_img.npy")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No files found for patient {patient_id}")
        
    # arrange slices in the correct order (0..155)
    files.sort()
    
    volume_slices = []
    mask_slices = []
    
    for f in files:
        # Load Image
        img = np.load(f) # (2, 240, 240)
        volume_slices.append(img)
        
        # Load Mask (Ground Truth) zum Vergleich
        mask_f = f.replace("_img.npy", "_mask.npy")
        mask = np.load(mask_f)
        mask_slices.append(mask)
        
    # Stacken: 2D Arrays --> 3D NumPy Arrays
    # Shape: (Slices, Channels, H, W) -> (155, 2, 240, 240)
    vol_numpy = np.stack(volume_slices)
    mask_numpy = np.stack(mask_slices)
    
    return vol_numpy, mask_numpy

def predict_volume(model, vol_numpy):
    """
    Pass volume through model (slice by slice)
    """
    print("Running Inference on 3D Volume...")
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(len(vol_numpy))):
            # Prepare Slice
            slice_img = vol_numpy[i] # (2, 240, 240)
            tensor = torch.from_numpy(slice_img).float().unsqueeze(0).to(DEVICE) # (1, 2, 240, 240)
            
            # Predict
            logits = model(tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()
            
            # Save result (CPU)
            predictions.append(pred_mask[0, 0].cpu().numpy())
            
    return np.stack(predictions) # (155, 240, 240)


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
    # 1. Load Model
    model = UNet(n_channels=2, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    
    # 2. Find a patient ID automatically
    sample_file = glob.glob(os.path.join(DATA_DIR, "*_img.npy"))[3000]
    filename = os.path.basename(sample_file)
    parts = filename.split('_')
    patient_id = "_".join(parts[:-2]) 
    
    # 3. Run Pipeline
    vol, mask = load_volume(patient_id)
    pred = predict_volume(model, vol)
    
    visualize(vol, mask, pred, patient_id)

if __name__ == "__main__":
    main()