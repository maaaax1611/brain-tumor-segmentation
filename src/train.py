import os
import argparse
import torch
import torch.nn as nn
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
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation (0.0-1.0)")
    
    # Model & Data
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "transformer"], help="Model architecture")
    parser.add_argument("--data_path", type=str, default=os.path.join('data', 'MICCAI_BraTS_2019_Data_Training'), help="Path to dataset")
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

def dice_loss(pred, target):
    return 1 - dice_coeff(pred, target)

def train_fn(loader, model, optimizer, scaler):
    model.train()
    loop = tqdm(loader, desc="Train")
    epoch_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.cuda.amp.autocast():
            predictions = torch.sigmoid(model(data))
            loss = dice_loss(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return epoch_loss / len(loader)

def validate_fn(loader, model):
    model.eval()
    dice_score = 0
    loop = tqdm(loader, desc="Val")
    
    with torch.no_grad():
        for data, targets in loop:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            predictions = torch.sigmoid(model(data))
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
    
    # Pin_memory=True is good for GPU, num_workers=0 is safest for Windows
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Model & Optimizer
    model = get_model(args.model).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    # 3. Training Loop
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_fn(train_loader, model, optimizer, scaler)
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