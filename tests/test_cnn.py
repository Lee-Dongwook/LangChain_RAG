import torch
from detector.models.simple_cnn import SimpleCNN

model = SimpleCNN(num_classes=2)
dummy = torch.randn(4, 3, 32, 32)
out = model(dummy)
print(out.shape)
