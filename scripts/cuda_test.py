import os
import torch

# Clear any existing CUDA device settings
os.environ.pop('CUDA_VISIBLE_DEVICES', None)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Get current device
    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")

    # Get device properties
    device_properties = torch.cuda.get_device_properties(current_device)
    print(f"Device properties: {device_properties}")

    # Try a simple CUDA operation
    x = torch.randn(3).cuda()
    print(f"Tensor device: {x.device}")
    y = x + 1
    print(f"Operation successful on {y.device}")