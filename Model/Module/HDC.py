import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Backbone.Xception65 import *
from Model.Module.CBAM import *


class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation):
        super(DWConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, BatchNorm=nn.BatchNorm2d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch)
        )
        self.ca = CBAM(in_ch=out_ch, no_spatial=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x_tmp = self.dwconv(x)
        x = self.conv1_2(x_tmp)
        x = self.ca(x)
        return x

class SCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, rates):
        super(SCBAM, self).__init__()
        self.dilated_conv = []
        for i, rate in enumerate(rates):
            tmp = DWConv(in_channels, out_channels // len(rates), kernel_size, stride, rate)
            self.dilated_conv.append(tmp)
        self.dilated_conv = nn.ModuleList(self.dilated_conv)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = []
        for hdc in self.dilated_conv:
            out.append(hdc(x))
        combined_features = torch.cat(out, 1)
        output = self.bn(combined_features)
        output = self.relu(output)
        return output

class HybridDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rates):
        super(HybridDilatedConv, self).__init__()

        effective_dilation = [rate * (kernel_size - 1) for rate in rates]
        p = [int((effective_dilation[i] - 1) / 2) for i in range(len(rates))]

        self.dilated_conv = []
        for i, rate in enumerate(rates):
            self.dilated_conv.append(nn.Conv2d(in_channels, out_channels // len(rates), kernel_size, padding=p[i]+1, dilation=rate, bias=False))
        self.dilated_conv = nn.ModuleList(self.dilated_conv)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = []
        for hdc in self.dilated_conv:
            out.append(hdc(x))
        # for i in out:
        #     print(i.size())
        combined_features = torch.cat(out, 1)
        output = self.bn(combined_features)
        output = self.relu(output)
        return output