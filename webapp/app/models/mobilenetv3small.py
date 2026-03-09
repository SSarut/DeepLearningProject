import torch
import torchvision.models as models
from types import MethodType
import torch.nn as nn

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        base = models.mobilenet_v3_small()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier
        self.classifier[-1] = torch.nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        x = self.classifier(f)
        if self.training:
            return x, f
        else:
            return x
