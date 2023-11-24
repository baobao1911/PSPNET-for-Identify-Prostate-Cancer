import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.excitation  = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduction_rate, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_planes // reduction_rate, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.excitation (self.avg_pool(x))
        max_out = self.excitation (self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        #self.conv_1 = nn.Conv2d(in_planes, in_planes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)*x
        x = self.sa(x)*x
        #x = self.conv_1(x)
        return x