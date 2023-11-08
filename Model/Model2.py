import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Xception_pretrained import AlignedXception
from Model.PPM import PPM
import math


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out

class MyModel(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.2, zoom_factor=16, n_classes=5, loss=nn.CrossEntropyLoss(ignore_index=255)):
        super(MyModel, self).__init__()
        assert 2048 % len(bins) == 0
        self.zoom_factor = zoom_factor
        self.criterion = loss#nn.CrossEntropyLoss()

        xception = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=zoom_factor)
        self.block = nn.Sequential(
            xception.conv1, xception.bn1, xception.relu
        )
        self.block0 = nn.Sequential(
            xception.conv2, xception.bn2, xception.relu
        )
        self.block1 = xception.block1
        self.block2_relu = xception.relu
        self.block2 = xception.block2
        self.block3 = xception.block3
        self.block4 = xception.block4
        self.block5 = xception.block5
        self.block6 = xception.block6
        self.block7 = xception.block7
        self.block8 = xception.block8
        self.block9 = xception.block9
        self.block10 = xception.block10
        self.block11 = xception.block11
        self.block12 = xception.block12
        self.block13 = xception.block13
        self.block14 = xception.block14
        self.block15 = xception.block15
        self.block16 = xception.block16
        self.block17 = xception.block17
        self.block18 = xception.block18
        self.block19 = xception.block19

        self.block20 = xception.block20

        self.block21 = nn.Sequential(
            xception.relu,

            xception.conv3,
            xception.bn3,
            xception.relu,
    
            xception.conv4,
            xception.bn4,
            xception.relu,

            xception.conv5,
            xception.bn5,
            xception.relu
        )
        self.dropout = nn.Dropout(p=dropout)

        self.relu = nn.ReLU(inplace=True)

        self.ppm = PPM(2048, int(2048/len(bins)), bins)
        
        self.conv_ppm = nn.Sequential(
            nn.Conv2d(2048*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            self.relu,
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1)
        )

        self.gau1 = GAU(512, 256, True)
        self.gau2 = GAU(256, 128, True)

        self.shallow_features = nn.Sequential(
            nn.Conv2d(128, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            self.relu,
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(128+48, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, n_classes, kernel_size=1),
        )

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.3),
                nn.Conv2d(512, n_classes, kernel_size=1),
            )

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

        
    def forward(self, x, y=None):
        _, _, h, w = x.size()
        x = self.block(x)
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x1 = self.block2_relu(x1)
        x2 = self.block2(x1)
        
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        x8 = self.block8(x7)
        x9 = self.block9(x8)
        x10 = self.block10(x9)
        x11 = self.block11(x10)
        x12 = self.block12(x11)
        x13 = self.block13(x12)
        x14 = self.block14(x13)
        x15 = self.block15(x14)
        x16 = self.block16(x15)
        x17 = self.block17(x16)
        x18 = self.block18(x17)
        x19 = self.block19(x18)
        x20 = self.block20(x19)
        x21 = self.block21(x20)

        x21 = self.dropout(x21)

        x_ppm = self.ppm(x21)
        x_ppm = self.conv_ppm(x_ppm)

        x_ppm_gau1 = self.gau1(x_ppm, x2)
        x_ppm_gau2 = self.gau2(x_ppm_gau1, x1)

        shallow_features = self.shallow_features(x1)

        x_cat = torch.cat([x_ppm_gau2, shallow_features], dim=1)

        x_cat = self.final_conv(x_cat)

        x_cat = F.interpolate(x_cat, size=(h, w), mode="bilinear", align_corners=True)

        if self.training:
            aux = self.aux(x20)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            aux_loss = self.criterion(aux, y)
            main_loss = self.criterion(x_cat, y)
            return x_cat.max(1)[1], main_loss, aux_loss
        else:
            return x_cat