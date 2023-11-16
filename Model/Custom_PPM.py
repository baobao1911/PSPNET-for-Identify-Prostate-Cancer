import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import Model.Resnet101 as models
from Model.PPM import PPM
from Model.Gau import GAU

class Custom_PPM(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.2, classes=6, zoom_factor=16, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(Custom_PPM, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        resnet_path = r'D:\University\MyProject\Source\Utils\resnet101-v2.pth'
        resnet = models.resnet101(pretrained=pretrained, model_path=resnet_path)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        self.atrous_conv = nn.ModuleList([])
        inch = 2048
        for rate in [1, 2, 5, 1, 2, 5]:
            self.atrous_conv.append(nn.Sequential(
                    nn.Conv2d(inch, 512, kernel_size=3, dilation=rate, padding=rate),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
            ))
            inch = 512
            
        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)

        self.cls = nn.Sequential(
            nn.Conv2d(7168, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(2048, 512, kernel_size=1)
        )

        self.gau1 = GAU(512, 512)
        # self.gau2 = GAU(512, 256, upsample=True)

        self.low_features = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(512+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, classes, kernel_size=1),
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

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        _, _, h4, w4 = x1.size()
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_atrous = x4
        if self.use_ppm:
            x = self.ppm(x4)

        atrous_layers = [x]
        for atrous_layer in self.atrous_conv:
            x_atrous = atrous_layer(x_atrous)
            atrous_layers.append(x_atrous)

        x  = torch.cat(atrous_layers, dim=1)
        x = self.cls(x)

        x = self.gau1(x, x2)
        
        x = F.interpolate(x, size=(h4, w4), mode='bilinear', align_corners=True)
        low_feature = self.low_features(x1)

        x = torch.cat([x, low_feature], dim=1)

        x = self.final_conv(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x3)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x