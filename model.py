# model.py
import torch.nn as nn
import torchvision.models as models


class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)