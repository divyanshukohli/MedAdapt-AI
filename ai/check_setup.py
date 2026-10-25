import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

print("MPS available:", torch.backends.mps.is_available())

x = torch.rand(3, 3)

if torch.backends.mps.is_available():
    x = x.to("mps")
    print("Apple GPU test:", x.device)
else:
    print("Using CPU for this test")

print("Setup successful!")