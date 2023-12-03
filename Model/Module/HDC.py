import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Backbone.Xception65 import *
from Model.Module.CBAM import *


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(out_channels, _make_divisible(in_channels // reduction, 8)),
                nn.SiLU(),
                nn.Linear(_make_divisible(in_channels // reduction, 8), out_channels),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    


class HDC_DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, rates):
        super(HDC_DWConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dilated_conv = nn.ModuleList([])
        for rate in rates:
            self.dilated_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=rate, dilation=rate, groups=out_channels, bias=False)
                    )
                )
            in_channels = out_channels

    def forward(self, x):
        conv = self.conv1(x)
        for hdc in self.dilated_conv:
            x = hdc(x)
        return torch.cat([x, conv], 1)

class HDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rates):
        super(HDC, self).__init__()
        self.dilated_conv = nn.ModuleList([])
        for rate in rates:
            self.dilated_conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
                )
            in_channels = out_channels

    def forward(self, x):
        for i, hdc in enumerate(self.dilated_conv):
            x = hdc(x)
        return x