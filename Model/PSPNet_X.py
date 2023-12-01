import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Backbone.Xception65 import *
from Model.Module.PPM import PPM
from Model.Module.HDC import *

class PSPNet_X(nn.Module):
    def __init__(self,bins=(1, 2, 3, 6), rates=[1, 2, 5], dropout=0.3, classes=6, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(PSPNet_X, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion

        model = xception(pretrained=True, model_path=r'xxx')
        self.layer0 = nn.Sequential(
            model.conv1, model.bn1, model.relu1,
            model.conv2, model.bn2, model.relu2
        )
        self.layer1 = model.block1
        self.layer2 = model.block2
        self.layer3 = model.block3
        self.layer4 = model.block4
        self.layer5 = model.block5
        self.layer6 = model.block6
        self.layer7 = model.block7
        self.layer8 = model.block8
        self.layer9 = model.block9
        self.layer10 = model.block10
        self.layer11 = model.block11
        self.layer12 = model.block12
        self.layer13 = nn.Sequential(
            x = model.conv3, x = model.bn3, x = model.relu3,
            x = model.conv4, x = model.bn4
        )

        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        
        self.hdc = HDC(fea_dim, 512, stride=1, rates=rates)
        fea_dim = fea_dim*2 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )
        self.shallow_feature = nn.Sequential(
            nn.Conv2d(128, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.segmentation = nn.Sequential(
            nn.Conv2d(512+48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        _, _, h, w = x.size()
        x = self.layer0(x)
        x_tmp = self.layer1(x)
        _, _, h4, w4 = x.size()
        shallow_features = self.shallow_feature(x_tmp)
        x = self.layer2(x_tmp)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x) 
        x = self.layer12(x)
        x = self.layer13(x)

        x_ppm = self.ppm(x)
        x_hdc = self.hdc(x)
        x = torch.cat([x_ppm, x_hdc], 1)
        x = self.cls(x)
        x = F.interpolate(x, size=(h4, w4), mode='bilinear', align_corners=True)
        x = torch.cat([x, shallow_features], 1)
        x = self.segmentation(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x