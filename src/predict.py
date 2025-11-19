import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from src.dataset import BraTSDataset

from src.models import UNet

# --- SETTINGS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join('data', 'processed_slices')
CHECKPOINT_DIR = "checkpoints"

# Try to load 'best_model.pth', fallback to 'last_model.pth'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')

def predict_and_show():
    print(f"--- INFERENCE DEBUG MODE ---")
    print(f"Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: No checkpoint found. Wait for the first epoch to finish.")
        return

    # 1. Load Model
    model = UNet(n_channels=2, n_classes=1).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.eval() # CRITICAL: Turns off Dropout and Batchnorm adjustment

    # 2. Load Dataset
    ds = BraTSDataset(DATA_PATH)
    
    # 3. Find a slice WITH a tumor
    # We loop until we find a mask that is not empty, so the visualization is interesting.
    print("Searching for a slice containing a tumor...")
    idx = 0
    found = False
    for _ in range(200): # Try 200 random slices
        rand_idx = random.randint(0, len(ds) - 1)
        _, mask = ds[rand_idx]
        if mask.max() > 0: # Found tumor!
            idx = rand_idx
            found = True
            break
    
    if not found:
        print("⚠️ Warning: Could not find a tumor slice in 200 tries. Showing random slice.")
        idx = random.randint(0, len(ds) - 1)

    print(f"Analyzing Slice Index: {idx}")
    image, mask = ds[idx]
    
    # 4. Inference
    input_tensor = image.unsqueeze(0).to(DEVICE) # Add batch dim: (1, 2, 240, 240)
    
    with torch.no_grad():
        # Get RAW Logits first
        logits = model(input_tensor)
        
        # Apply Sigmoid to get Probabilities (0.0 to 1.0)
        pred_prob = torch.sigmoid(logits)
        
        # Create Binary Mask (Threshold 0.5)
        pred_mask = (pred_prob > 0.5).float()

    # --- DIAGNOSTICS (The important part) ---
    max_val = pred_prob.max().item()
    mean_val = pred_prob.mean().item()
    min_val = pred_prob.min().item()
    
    print("-" * 30)
    print(f"🔍 MODEL CONFIDENCE REPORT:")
    print(f"   Max Probability in Image: {max_val:.6f}")
    print(f"   Avg Probability in Image: {mean_val:.6f}")
    print(f"   Min Probability in Image: {min_val:.6f}")
    
    if max_val < 0.01:
        print("👉 Diagnosis: Model is 'Dead' (Predicting almost pure zero). Needs more training or BCE Loss.")
    elif max_val < 0.5:
        print("👉 Diagnosis: Model is 'Shy'. It sees something, but confidence is below 50%. Keep training!")
    else:
        print("👉 Diagnosis: Model is Active! It is predicting tumor pixels.")
    print("-" * 30)

    # 5. Visualization
    # Move to CPU for plotting
    img_numpy = image[0].cpu().numpy() # Show Channel 0 (FLAIR)
    mask_numpy = mask[0].cpu().numpy()
    prob_numpy = pred_prob[0, 0].cpu().numpy() # Show the raw probability map
    pred_numpy = pred_mask[0, 0].cpu().numpy() # Show the binary result

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(img_numpy, cmap='gray')
    ax[0].set_title("Input (FLAIR)")
    ax[0].axis('off')
    
    ax[1].imshow(mask_numpy, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')
    
    # Visualizing the 'Heatmap' (Confidence)
    im2 = ax[2].imshow(prob_numpy, cmap='jet', vmin=0, vmax=1)
    ax[2].set_title(f"Model Confidence\n(Max: {max_val:.4f})")
    ax[2].axis('off')
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    
    ax[3].imshow(pred_numpy, cmap='gray')
    ax[3].set_title("Binary Prediction (> 0.5)")
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_and_show()