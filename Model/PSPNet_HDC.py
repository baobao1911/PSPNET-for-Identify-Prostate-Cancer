import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Backbone.Resnet101 import *
from Model.Module.HDC import *
from Model.Module.PPM import *


class PSPNet_HDC(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), rates=[1, 2, 5, 1 ,2, 5], dropout=0.25, classes=6, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSPNet_HDC, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion

        resnet_path = r'D:\University\Semantic_Segmentation_for_Prostate_Cancer_Detection\Semantic_Segmentation_for_Prostate_Cancer_Detection\Utils\resnet101-v2.pth'
        resnet = resnet101(pretrained=pretrained, model_path=resnet_path)
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

        fea_dim = 2048
        self.ppm = PPM_custom(fea_dim, int(fea_dim/len(bins)), bins, rates)


        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        fea_dim = int(fea_dim/len(bins)) + 48
        self.pooling= PPM(fea_dim, int(fea_dim/len(bins)), bins)

        fea_dim *=2
        self.fc = nn.Sequential(
                nn.Conv2d(fea_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
        
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        _, _, h, w = x.size()

        x = self.layer0(x)
        x_tmp = self.layer1(x)
        reduce = self.reduce(x_tmp)

        x = self.layer2(x_tmp)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.ppm(x)
        x = F.interpolate(x, size=reduce.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, reduce], 1)
        x = self.pooling(x)

        x = self.fc(x)

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