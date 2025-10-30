import torch.nn as nn, torchvision
import pandas as pd
import numpy as np

class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), # 16 * 224 * 224
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 * 112 * 112
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 32 * 56 * 56
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ZFnet(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.pool6 = nn.AdaptiveAvgPool2d((6, 6))

        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size= 7, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2),
            nn.Conv2d(96, 256, kernel_size= 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha = 1e-4, beta = 0.75, k = 2.0),
            nn.MaxPool2d(3, stride = 2),
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool6(x)
        x = self.classifier(x)
        return x
