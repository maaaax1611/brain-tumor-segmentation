import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F  # WICHTIG: Für die Loss-Funktion
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import N_CHANNELS, N_CLASSES, PROCESSED_DIR, SPLIT_FILE, CHECKPOINT_DIR
from src.dataset import BraTSDataset
from src.metrics import evaluate_patients, mean_per_region
from src.models import UNet
from src.splits import get_or_create_split

# Global Device Setting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Segmentation Model")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of PATIENTS used for validation")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of PATIENTS held out for testing")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the patient split")
    parser.add_argument("--split_file", type=str, default=SPLIT_FILE, help="Patient split JSON (created if missing)")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model & Data
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "transformer"], help="Model architecture")
    parser.add_argument("--data_path", type=str, default=PROCESSED_DIR, help="Path to preprocessed .npy files")
    parser.add_argument("--save_dir", type=str, default=CHECKPOINT_DIR, help="Directory to save model checkpoints")
    
    return parser.parse_args()

def get_model(model_name):
    """Factory function to get the correct model architecture."""
    if model_name == "unet":
        return UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    elif model_name == "transformer":
        raise NotImplementedError("Transformer implementation coming soon!")
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def criterion(logits, targets, dice_weight=5.0):
    """
    Combined BCE + Dice loss over the region channels (WT, TC, ET).
    Takes RAW LOGITS (before sigmoid) for better numerical stability.
    Dice is computed per region over the batch and then averaged, so the small
    regions (enhancing tumor) weigh as much as the whole tumor.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    smooth = 1.0
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dims)
    dice = (2.0 * intersection + smooth) / (probs.sum(dims) + targets.sum(dims) + smooth)

    return bce + dice_weight * (1 - dice.mean())

def train_fn(loader, model, optimizer):
    model.train()
    loop = tqdm(loader, desc="Train")
    epoch_loss = 0
    valid_batches = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(data)            
        loss = criterion(logits, targets)

        if torch.isnan(loss):
            loop.set_postfix(loss="NaN (Skipped)")
            continue

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        epoch_loss += loss.item()
        valid_batches += 1
        loop.set_postfix(loss=loss.item())
    
    if valid_batches == 0: return 0.0
    return epoch_loss / valid_batches

def validate_fn(model, data_path, patient_ids):
    """Mean Dice per region over validation PATIENTS, computed on full 3D volumes."""
    return mean_per_region(evaluate_patients(model, data_path, patient_ids, DEVICE))

def main():
    args = parse_args()
    
    print(f"--- Setup: Model={args.model} | Device={DEVICE} | Batch={args.batch_size} | LR={args.lr} ---")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Prepare Data (patient-level split -> no slice leakage between sets)
    split = get_or_create_split(args.data_path, args.split_file, args.val_split, args.test_split, args.seed)
    train_ds = BraTSDataset(args.data_path, split["train"])
    val_patients = split["val"]
    print(f"Train: {len(train_ds)} slices | Val: {len(val_patients)} patients | "
          f"Test (held out): {len(split['test'])} patients")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    # 2. Model & Optimizer
    model = get_model(args.model).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_fn(train_loader, model, optimizer)
        val_regions = validate_fn(model, args.data_path, val_patients)
        val_dice = sum(val_regions.values()) / len(val_regions)  # model selection: mean over regions

        region_str = " | ".join(f"{k} {v:.4f}" for k, v in val_regions.items())
        print(f"Result: Train Loss: {train_loss:.4f} | Val Dice (3D, per patient): {region_str} | mean {val_dice:.4f}")
        
        # Save Checks
        last_path = os.path.join(args.save_dir, "last_model.pth")
        torch.save(model.state_dict(), last_path)
        
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🔥 New Best Model saved! (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    main()