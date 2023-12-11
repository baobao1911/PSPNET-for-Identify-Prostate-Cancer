import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Module.CBAM import *

class SeparableConv2d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self. bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels, reduction_ratio=4)
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        return x

class PPM_custom(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, rates):
        super(PPM_custom, self).__init__()
        self.protect_x = nn.Sequential(
                CBAM(in_dim, reduction_ratio=4),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False)
        )

        self.pooling = nn.ModuleList([])
        for bin in bins:
            self.pooling.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                CBAM(in_dim, reduction_ratio=4),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ))
        self.hybrid_dilation = nn.ModuleList([])
        in_channels = in_dim
        for rate in rates:
            self.hybrid_dilation.append(SeparableConv2d(in_channels, reduction_dim, kernel_size=3, stride=1, padding=rate, dilation=rate))
            in_channels = reduction_dim
        
        self.project = nn.Sequential(
            nn.Conv2d(reduction_dim*(len(bins) + 2), reduction_dim, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
            )

    def forward(self, x):
        x_size = x.size()
        out = [self.protect_x(x)]
        for f in self.pooling:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        hdc = x
        for hd in self.hybrid_dilation:
            hdc = hd(hdc)
        out.append(hdc)
        output = torch.cat(out, 1)
        return self.project(output)




class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        
        self.features = nn.ModuleList([])
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ))
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)