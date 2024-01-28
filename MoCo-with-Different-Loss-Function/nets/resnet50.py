#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], use_shortcut=False):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        self.use_shortcut = use_shortcut
        self.shortcut = nn.Sequential()
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        if self.use_shortcut:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, Bottleneck):
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        flag = True
        for i in range(len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], use_shortcut=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)

        return out

