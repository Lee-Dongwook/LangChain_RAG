import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
