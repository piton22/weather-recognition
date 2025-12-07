import torch.nn as nn
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Заморозка большей части сети
        for p in self.model.parameters():
            p.requires_grad = False

        # Разморозка последних блоков
        for p in self.model.layer3.parameters():
            p.requires_grad = True
        for p in self.model.layer4.parameters():
            p.requires_grad = True

        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)