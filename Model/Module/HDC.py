import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Backbone.Xception65 import *
from Model.Module.CBAM import *


class DWConv(nn.Module):
    def __init__(self, in_ch, kernel_size, stride, dilation):
        super(DWConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, int(in_ch//4), kernel_size=1),
            nn.BatchNorm2d(int(in_ch//4)),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(
            SeparableConv2d(int(in_ch//4), int(in_ch//4), kernel_size=kernel_size, stride=stride, dilation=dilation, BatchNorm=nn.BatchNorm2d),
            nn.BatchNorm2d(int(in_ch//4)),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(int(in_ch//4), in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch)
        )
        self.ca = CBAM(in_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_tmp = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv1_2(x)
        x = self.ca(x) + x_tmp
        x = self.relu(x)
        return x

class SCBAM(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, rates):
        super(SCBAM, self).__init__()
        self.dilated_conv = []
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//4), kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )

        for i, rate in enumerate(rates):
            tmp = DWConv(int(in_channels//4), kernel_size, stride, rate)
            self.dilated_conv.append(tmp)
        self.dilated_conv = nn.ModuleList(self.dilated_conv)

        self.bn = nn.BatchNorm2d(int(in_channels//4))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dwconv(x)
        for hdc in self.dilated_conv:
            x = hdc(x)*x

        output = self.bn(x)
        output = self.relu(output)
        return output

class HybridDilatedConv(nn.Module):
    def __init__(self, in_channels, kernel_size, rates):
        super(HybridDilatedConv, self).__init__()

        effective_dilation = [rate * (kernel_size - 1) for rate in rates]
        p = [int((effective_dilation[i] - 1) / 2) for i in range(len(rates))]
        channel = int(in_channels//4)

        self.dilated_conv = []
        for i, rate in enumerate(rates):
            self.dilated_conv.append(nn.Conv2d(in_channels, channel, kernel_size, padding=p[i]+1, dilation=rate, bias=False))
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