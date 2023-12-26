import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Backbone.Resnet import *
from Model.Module.PPM import *
from Model.Module.Gau import *

class PSPNet_Custom(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), rates=[[1, 6, 12, 18],[1, 2, 5, 1, 2, 5]], 
                 dropout=0.25, classes=6, zoom_factor=16, 
                 criterion=nn.CrossEntropyLoss(ignore_index=255), 
                 pretrained=True, Backbone_path='Utils'):
        super(PSPNet_Custom, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        self.zoom_factor = zoom_factor
        self.criterion = criterion

        resnet = resnext50_32x4d(pretrained=pretrained, model_path=Backbone_path)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)#(4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.ppm = PPM_Custom(fea_dim, int(fea_dim/len(bins)), bins, rates[0], use_SeparableConv=True, add_hdc=False, add_asp=True)

        fea_dim = int(fea_dim/len(bins))

        self.gau1 = GAU(fea_dim, 512)
        self.gau2 = GAU(512, 256)

        self.fc = nn.Sequential(
               nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x = self.ppm(x4) #/16
        x = self.gau1(x, x2) #/8
        x = self.gau2(x, x1) #/4

        x = self.fc(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x1)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x