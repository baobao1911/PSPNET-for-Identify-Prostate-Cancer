import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Xception import Xception, SeparableConv2d
from Model.PPM import PPM
from Model.FPM import SegmentationBlock, FPNBlock


class MyModel(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.2, zoom_factor=8, n_classes=5, loss=nn.CrossEntropyLoss(ignore_index=255)):
        super(MyModel, self).__init__()
        assert 2048 % len(bins) == 0
        self.zoom_factor = zoom_factor
        self.criterion = loss #nn.CrossEntropyLoss(weight=torch.tensor([2.05586943, 0.49666363, 0.98972499,  0.70038494, 16.13300493]), ignore_index=255)

        xception = Xception(zoom_factor=zoom_factor)
        
        self.block = nn.Sequential(
            xception.conv1, xception.bn1, xception.relu
        )
        self.block0 = nn.Sequential(
            xception.conv2, xception.bn2, xception.relu
        )
        self.block1 = xception.block1
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
            xception.conv3,
            xception.conv4,
            xception.conv5
        )

        self.relu = nn.ReLU(inplace=True)
        self.shallow_features = nn.Sequential(
            SeparableConv2d(256, 48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            self.relu,
        )

        self.ppm = PPM(2048, int(2048/len(bins)), bins)
        self.conv_ppm = nn.Sequential(
            nn.Conv2d(2048*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            self.relu,
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(560, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, n_classes, kernel_size=1),
        )

        self.dropout = nn.Dropout(p=dropout)
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(128, n_classes, kernel_size=1),
            )



    def forward(self, x, y=None):
        _, _, h, w = x.size()
        x = self.block(x)
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        shallow_features = self.block2.hook_layer
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


        x_ppm = self.ppm(x21)
        x_ppm = self.dropout(x_ppm)
        #print(x_ppm.size())
        x_ppm = self.conv_ppm(x_ppm)
        #print(x_ppm.size())

        x_ppm = F.interpolate(x_ppm, scale_factor=2, mode="bilinear", align_corners=True)
        #print(x_ppm.size())
        shallow_features = self.shallow_features(shallow_features)
        #print(shallow_features.size())
        x_cat = torch.cat([x_ppm, shallow_features], dim=1)
        x_cat = self.final_conv(x_cat)

        x_cat = F.interpolate(x_cat, scale_factor=4, mode="bilinear", align_corners=True)
        if self.training:
            aux = self.aux(x2)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            aux_loss = self.criterion(aux, y)
            main_loss = self.criterion(x_cat, y)
            #print('\n', main_loss.item())
            return x_cat.max(1)[1], main_loss, aux_loss
        else:
            return x_cat