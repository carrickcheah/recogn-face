import torch
import torch.nn as nn
from torchvision import models

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a model (e.g., ResNet18)
model = models.resnet18(pretrained=True)

# Move the model to the device (GPU/CPU)
model = model.to(device)
