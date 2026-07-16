"""
resnet3d.py
===========
3D ResNet-18 backbone extracted from the HOPE Implementation.
Produces a 128-dimensional feature vector per subject.
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet3D', 'resnet18_3d']


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4),
        device=out.device, dtype=out.dtype,
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet backbone (prototype-free).
    The forward() returns a single 128-dimensional feature vector.
    """

    def __init__(
        self,
        block,
        layers,
        spatial_size=128,
        sample_duration=128,
        shortcut_type='B',
    ):
        self.inplanes = 64
        super().__init__()

        # ── Stem ────────────────────────────────────────────────────────────
        self.conv1 = nn.Conv3d(
            1, 64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1   = nn.BatchNorm3d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ── Residual stages ─────────────────────────────────────────────────
        self.layer1 = self._make_layer(block, 64,  layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        # ── Global average pool ─────────────────────────────────────────────
        last_duration = int(math.ceil(sample_duration / 32))
        last_size     = int(math.ceil(spatial_size    / 32))
        self.avgpool  = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        # ── Feature projector (same as HOPE fc1: 512 -> 128) ────────────────
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
        )

        # ── Weight initialisation (Kaiming, matching original HOPE code) ────
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ────────────────────────────────────────────────────────────────────────
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        Input
        -----
        x : (B, 1, 128, 128, 128)

        Returns
        -------
        features : (B, 128)  — 128-dim feature vector
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # (B, 512)
        x = self.fc1(x)              # (B, 128)
        return x


def resnet18_3d(**kwargs) -> ResNet3D:
    """Constructs a prototype-free 3D ResNet-18 backbone."""
    return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
