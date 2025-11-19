import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F  # WICHTIG: Für die Loss-Funktion
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Imports
from src.dataset import BraTSDataset
from src.models import UNet

# Global Device Setting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Segmentation Model")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")
    
    # Model & Data
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "transformer"], help="Model architecture")
    parser.add_argument("--data_path", type=str, default=os.path.join('data', 'processed_slices'), help="Path to preprocessed .npy files")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    
    return parser.parse_args()

def get_model(model_name):
    """Factory function to get the correct model architecture."""
    if model_name == "unet":
        return UNet(n_channels=2, n_classes=1)
    elif model_name == "transformer":
        raise NotImplementedError("Transformer implementation coming soon!")
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def dice_coeff(pred, target):
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def criterion(logits, targets):
    """
    Combined BCE + Dice Loss.
    Takes RAW LOGITS (before sigmoid) for better numerical stability.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    
    probs = torch.sigmoid(logits)
    smooth = 1.0
    
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (probs_flat * targets_flat).sum()
    
    dice_score = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    dice_loss = 1 - dice_score
    
    return bce + dice_loss

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

def validate_fn(loader, model):
    model.eval()
    dice_score = 0
    loop = tqdm(loader, desc="Val")
    
    with torch.no_grad():
        for data, targets in loop:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            logits = model(data)
            predictions = torch.sigmoid(logits)
            
            predictions = (predictions > 0.5).float()
            dice_score += dice_coeff(predictions, targets).item()
            
    return dice_score / len(loader)

def main():
    args = parse_args()
    
    print(f"--- Setup: Model={args.model} | Device={DEVICE} | Batch={args.batch_size} | LR={args.lr} ---")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Prepare Data
    print(f"Loading data from: {args.data_path}")
    full_dataset = BraTSDataset(args.data_path)
    
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    print(f"Data Split: {train_size} Train | {val_size} Val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model & Optimizer
    model = get_model(args.model).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_fn(train_loader, model, optimizer)
        val_dice = validate_fn(val_loader, model)
        
        print(f"Result: Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
        
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