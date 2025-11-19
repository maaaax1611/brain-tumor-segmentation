import os
import torch
from src.dataset import BraTSDataset

def test_data_loading():
    # 1. Define the path to your unzipped data
    data_path = os.path.join('data', 'MICCAI_BraTS_2019_Data_Training')
    
    print(f"Looking for data in: {os.path.abspath(data_path)}")

    if not os.path.exists(data_path):
        print("ERROR: Path does not exist. Please check your folder structure.")
        return

    # 2. Initialize the Dataset
    print("Initializing Dataset...")
    try:
        ds = BraTSDataset(data_path)
    except Exception as e:
        print(f"ERROR initializing dataset: {e}")
        return

    if len(ds) == 0:
        print("ERROR: Dataset found no patients. Check if 'HGG' and 'LGG' folders are inside the data path.")
        return

    print(f"SUCCESS: Found {len(ds)} total slices across all patients.")

    # 3. Load one sample (Test the __getitem__ method)
    print("Loading one sample (this might take a second)...")
    try:
        # We pick a random index, e.g., 1000, to hit a real slice
        image, mask = ds[1000] 
        
        print("-" * 30)
        print("DATA LOADED SUCCESSFULLY!")
        print(f"Input Image Shape: {image.shape} (Expected: [2, 240, 240])")
        print(f"Target Mask Shape: {mask.shape}  (Expected: [1, 240, 240])")
        print(f"Data Type: {image.dtype}")
        print(f"Max Value in Image: {image.max():.2f} (Should be small due to normalization)")
        print("-" * 30)
        
    except Exception as e:
        print(f"ERROR loading sample: {e}")

if __name__ == "__main__":
    test_data_loading()