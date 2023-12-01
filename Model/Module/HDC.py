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
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, rate):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_channels * expand_ratio)
        self.identity = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=rate, dilation=rate, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            
            SELayer(in_channels, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class HDC_MBConv(nn.Module):
    def __init__(self, in_channels, stride, rates):
        super(HDC_MBConv, self).__init__()
        self.dilated_conv = []
        for rate in rates:
            tmp = MBConv(in_channels, in_channels, stride, 0.2, rate)
            self.dilated_conv.append(tmp)
        self.dilated_conv = nn.ModuleList(self.dilated_conv)

    def forward(self, x):
        for hdc in self.dilated_conv:
            x = hdc(x)
        return x

class HybridDilatedConv(nn.Module):
    def __init__(self, in_channels, kernel_size, rates):
        super(HybridDilatedConv, self).__init__()


        channel = int(in_channels//4)

        self.dilated_conv = []
        for i, rate in enumerate(rates):
            self.dilated_conv.append(nn.Conv2d(in_channels, channel, kernel_size, padding=rate, dilation=rate, bias=False))
            in_channels = channel
        self.dilated_conv = nn.ModuleList(self.dilated_conv)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        outs = []
        for i, hdc in enumerate(self.dilated_conv):
            x = hdc(x)
            outs.append(x)
        x = outs[0]
        for i in range(1, len(outs)):
            x *=outs[i]

        output = self.bn(x)
        output = self.relu(output)
        return output