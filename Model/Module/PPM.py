import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Module.CBAM import *


class SeparableConv2d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, reduction_ratio=16):
        super(SeparableConv2d, self).__init__()
        self.cbam = CBAM(in_channels, reduction_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x = self.cbam(x)
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.dropout(x)
        return x
    
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super(Conv2D, self).__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x
    

# class PPM_custom(nn.Module):
#     def __init__(self, in_dim, reduction_dim, bins, rates):
#         super(PPM_custom, self).__init__()

#         self.pooling = nn.ModuleList([])
#         for bin in bins:
#             self.pooling.append(nn.Sequential(
#                 CBAM(in_dim, reduction_ratio=16),
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(reduction_dim),
#                 nn.ReLU(inplace=True),
#             ))

#         self.hybrid_dilation = nn.ModuleList([])
#         in_channels = in_dim
#         for rate in rates:
#             self.hybrid_dilation.append(SeparableConv2d(in_channels, reduction_dim, kernel_size=3, stride=1, padding=rate, dilation=rate))
#             in_channels = reduction_dim
        
#         self.cls = nn.Sequential(
#             CBAM(reduction_dim*(len(bins) + 1)+ in_dim, reduction_ratio=16),
#             nn.Conv2d(reduction_dim*(len(bins) + 1) + in_dim, reduction_dim, 1, bias=False),
#             nn.BatchNorm2d(reduction_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2)
#             )
#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.pooling:
#             out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
#         hdc = x
#         for hd in self.hybrid_dilation:
#             hdc = hd(hdc)
#         out.append(hdc)
#         output = torch.cat(out, 1)
#         return self.cls(output)

class PPM_Custom(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, rates, use_SeparableConv=False, add_hdc=False, add_asp=False, attention=True):
        super(PPM_Custom, self).__init__()

        if add_asp == True and add_hdc == True:
            assert False , "Both add_asp and add_hdc cannot be the same at the same time as True"

        self.hdc = add_hdc
        self.asp = add_asp
        self.use_SeparableConv = use_SeparableConv
        
        self.skip = False
        if add_asp or add_hdc:
            self.skip = True

        self.pooling = nn.ModuleList([])
        for bin in bins:
            pool = []
            if attention:
                pool.append(CBAM(in_dim,reduction_ratio=16))
            pool.append(nn.AdaptiveAvgPool2d(bin))
            pool.append(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False))
            pool.append(nn.BatchNorm2d(reduction_dim))
            pool.append(nn.ReLU(inplace=True))
            self.pooling.append(nn.Sequential(*pool))
        decode_dim = reduction_dim*len(bins)
        
        if self.hdc:
            self.hdc = nn.ModuleList([])
            self.hdc.append(Conv2D(in_dim, reduction_dim, kernel_size=3, padding=1, dilation=1))
            if self.use_SeparableConv:
                for rate in rates[1:]:
                    self.hdc.append(SeparableConv2d(reduction_dim, reduction_dim, kernel_size=3, stride=1, padding=rate, dilation=rate, reduction_ratio=4))
            else:
                for rate in rates[1:]:
                    self.hdc.append(Conv2D(reduction_dim, reduction_dim, kernel_size=3, padding=rate, dilation=rate))
            decode_dim += reduction_dim

        if self.asp:
            self.asp = nn.ModuleList([])
            if use_SeparableConv == False:
                for rate in rates[1:]:
                    self.asp.append(Conv2D(in_dim, reduction_dim, kernel_size=3, padding=rate, dilation=rate))
            else:
                for rate in rates[1:]:
                    self.asp.append(SeparableConv2d(in_dim, reduction_dim, kernel_size=3, stride=1, padding=rate, dilation=rate, reduction_ratio=16))
            decode_dim += reduction_dim*(len(rates)-1)

        if self.skip:
            self.skip_x = nn.Sequential(
                CBAM(in_dim, reduction_ratio=16),
                Conv2D(in_dim, reduction_dim, kernel_size=1, padding=0, dilation=1, bias=False))
            decode_dim +=reduction_dim
        else:
            decode_dim +=in_dim

        self.cls = nn.Sequential(
            CBAM(decode_dim, reduction_ratio=16),
            Conv2D(decode_dim, reduction_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.Dropout(0.3)
            )

    def forward(self, x):
        x_size = x.size()
        out = []
        if self.skip:
            out.append(self.skip_x(x))
        else:
            out.append(x)

        for f in self.pooling:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))

        if self.hdc:
            for doSomething in self.hdc:
                x = doSomething(x)
            out.append(x)

        if self.asp:
            for doSomething in self.asp:
                out.append(doSomething(x))

        output = torch.cat(out, 1)
        return self.cls(output)

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