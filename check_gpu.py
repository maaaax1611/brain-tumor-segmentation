import torch
import sys

def check_system():
    """
    Simple script to verify PyTorch installation and CUDA availability within Conda.
    """
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"CUDA Available: YES (Version {torch.version.cuda})")
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        
        # Create a random tensor on GPU to test memory allocation
        try:
            x = torch.rand(5, 3).cuda()
            print("Success: Tensor created on GPU.")
        except Exception as e:
            print(f"Error: Could not create tensor on GPU. {e}")
    else:
        print("WARNING: CUDA is NOT available. Check your conda install command.")

if __name__ == "__main__":
    check_system()