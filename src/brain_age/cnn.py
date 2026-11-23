"""Compact 3D CNN for brain-age regression (input 1 x 80 x 96 x 80, 2 mm).

Five blocks of [3x3x3 conv -> batch norm -> ReLU -> 2x2x2 max pool], channels doubling from 16,
then global average pooling, dropout and one linear layer giving one number (scaled age).
About 1.2 million weights: small enough to train on ~1,750 scans without heavy overfitting,
in the spirit of SFCN (Peng et al., 2021) but simpler. It is a standard design, not a
contribution of this project (PROJECT_CONTEXT section 8).

Spatial size per block: 80x96x80 -> 40x48x40 -> 20x24x20 -> 10x12x10 -> 5x6x5 -> 2x3x2.
"""
import torch
from torch import nn


class BrainAgeCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128, 256), dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv3d(c_in, c, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm3d(c), nn.ReLU(inplace=True), nn.MaxPool3d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(1)


def count_weights(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
