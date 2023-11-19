import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rates=[1, 2, 5, 1, 2, 5]):
        super(HybridDilatedConv, self).__init__()
        self.dilated_conv = []
        for rate in rates:
            self.dilated_conv.append(nn.Conv2d(in_channels, out_channels // 6, kernel_size, padding=rate, dilation=rate, bias=False))
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
    
    
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.feature_base = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
        )
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        x1 = self.feature_base(x)
        out = [x1]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)