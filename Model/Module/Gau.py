import torch
import torch.nn as nn
from Model.Module.CBAM import *


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.low = nn.Sequential(
                CBAM(channels_low, reduction_ratio=16),
                nn.Conv2d(channels_low, channels_low, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels_low),
        )


        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)


        self.conv_high = nn.Sequential(
            CBAM(channels_high, reduction_ratio=16),
            nn.Conv2d(channels_high, channels_low, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_low),
        )


    def forward(self, fms_high, fms_low):
        fms_high_gp = self.global_pooling(fms_high)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low_mask = self.low(fms_low)
        fms_low_mask = self.relu(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp


        channels_high_upsampling = F.interpolate(fms_high, size=fms_low.size()[2:], mode='bilinear', align_corners=True)
        channels_high_upsampling = self.conv_high(channels_high_upsampling)
        channels_high_upsampling = self.relu(channels_high_upsampling)

        return channels_high_upsampling + fms_att
    

