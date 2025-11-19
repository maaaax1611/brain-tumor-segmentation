import os
import torch
import numpy as np
import pyvista as pv
import glob
from tqdm import tqdm
from src.models import UNet

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join('data', 'processed_slices')
CHECKPOINT = os.path.join('checkpoints', 'best_model.pth')

def load_volume(patient_id):
    """
    Lädt alle Slices eines Patienten und stapelt sie zu einem 3D-Volumen.
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
    Jagt das Volumen Slice für Slice durch das Modell.
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

def visualize(vol, mask, pred):
    """
    Create interactive 3d rendering.
    """
    print("Rendering 3D Scene... (Window will pop up)")
    
    # FLAIR-Channel (Channel 0) contains brain structure
    brain_vol = vol[:, 0, :, :] 
    
    # create PyVista Grid
    grid = pv.wrap(brain_vol)
    
    p = pv.Plotter(shape=(1, 2)) # 2 Windows: ground truth vs prediction
    
    # --- Window 1: Ground Truth ---
    p.subplot(0, 0)
    p.add_text("Ground Truth (Arzt)", font_size=12)
    
    # 1. Brain (Transparent)
    p.add_volume(brain_vol, cmap="bone", opacity="sigmoid", shade=True)
    
    # 2. Tumor (red)
    # create grid just for mask
    grid_mask = pv.wrap(mask[:, 0, :, :]) # Mask has Shape (155, 1, 240, 240) -> Squeeze
    # Threshold: Only show voxel with val=1
    tumor_mesh = grid_mask.threshold(0.5)
    p.add_mesh(tumor_mesh, color="lime", opacity=1.0, label="Tumor GT")

    # --- Window 2: Prediction ---
    p.subplot(0, 1)
    p.add_text(f"AI Prediction (Dice: ?)", font_size=12)
    
    # 1. Brain
    p.add_volume(brain_vol, cmap="bone", opacity="sigmoid", shade=True)
    
    # 2. Tumor (blue)
    grid_pred = pv.wrap(pred)
    pred_mesh = grid_pred.threshold(0.5)
    p.add_mesh(pred_mesh, color="red", opacity=1.0, label="AI Prediction")
    
    p.link_views()
    p.show()

def main():
    # 1. Load Model
    model = UNet(n_channels=2, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    
    # 2. Find a patient ID automatically
    sample_file = glob.glob(os.path.join(DATA_DIR, "*_img.npy"))[0]
    filename = os.path.basename(sample_file)
    parts = filename.split('_')
    patient_id = "_".join(parts[:-2]) 
    
    # 3. Run Pipeline
    vol, mask = load_volume(patient_id)
    pred = predict_volume(model, vol)
    
    visualize(vol, mask, pred)

if __name__ == "__main__":
    main()