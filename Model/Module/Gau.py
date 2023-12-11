import torch
import torch.nn as nn
from Model.Module.CBAM import *


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        #self.cbam = CBAM(channels_high, reduction_ratio=2)
        self.conv_high = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):

        fms_high_gp = self.global_pooling(fms_high)#nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp

        #out = self.cbam(fms_high)
        out = self.relu(self.bn(self.conv_high(fms_high)))
        out = F.interpolate(out, size=fms_att.size()[2:], mode='bilinear', align_corners=True)
        out = out + fms_att
        return out
    


class GAU_Custom(nn.Module):
    def __init__(self, channels_high, channels_low, bins):
        super(GAU_Custom, self).__init__()
        # Global Attention Upsample
        self.conv1x1_low = nn.Sequential(
            CBAM(channels_low),
            nn.Conv2d(channels_low, channels_low, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_low),
            nn.ReLU(inplace=True),
        )

        self.gpooling = nn.ModuleList([])
        reduction_dim = int(channels_high/len(bins))
        for bin in bins:
            self.gpooling.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                CBAM(channels_high, reduction_ratio=4),
                nn.Conv2d(channels_high, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ))
        self.conv1 = nn.Sequential(
            CBAM(channels_high, reduction_ratio=4),
            nn.Conv2d(channels_high, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )
        self.conv1x1_high = nn.Sequential(
                nn.Conv2d(reduction_dim*(len(bins)+ 1), channels_low, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels_low),
                nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            CBAM(channels_low*2, reduction_ratio=4),
            nn.Conv2d(channels_low*2, channels_low, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_low),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, high, low):

        high_branchs = [self.conv1(high)]
        for f in self.gpooling:
            high_branchs.append(F.interpolate(f(high), high.size()[2:], mode='bilinear', align_corners=True))
        high_branchs = torch.cat(high_branchs, 1)
        high_branchs = self.conv1x1_high(high_branchs)

        high_branchs = F.interpolate(high_branchs, low.size()[2:], mode='bilinear', align_corners=True)

        low_branchs = self.conv1x1_low(low)

        out = self.conv_out(torch.cat([low_branchs, high_branchs], 1))
        
        return out