import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Backbone.Resnet101 import *
from Model.Module.HDC import *
from Model.Module.PPM import *
from Model.Backbone.Xception65 import *
from Model.Module.CBAM import *
from Model.Module.DAModule import *

class DWConv(nn.Module):
    def __init__(self, in_ch, stride):
        super(DWConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(
            SeparableConv2d(in_ch, in_ch, kernel_size=3, stride=stride, BatchNorm=nn.BatchNorm2d),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch)
        )

        self.ca = nn.Sequential(
            ChannelAttention(in_planes=in_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv1_2(x)
        x = self.ca(x)*x
        return x
        
        


class PSPNet_HDC(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), rates=[1, 2, 5, 1, 2, 5], dropout=0.3, classes=6, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
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

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        fea_dim_hdc = 256*len(rates)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.hdc = HybridDilatedConv(fea_dim, fea_dim_hdc, kernel_size=3, rates=rates)

        fea_dim = fea_dim*2 + fea_dim_hdc
        self.da = nn.Sequential(
                nn.Dropout2d(p=dropout),
                DAModule(fea_dim, 512, norm_layer=nn.BatchNorm2d),
            )
        self.cls = nn.Sequential(
            nn.Conv2d(512, classes, kernel_size=1)
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
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        x_ppm = self.ppm(x)
        x_hdc = self.hdc(x)
        x = torch.cat([x_hdc, x_ppm], dim=1)
        x = self.da(x)
        x = self.cls(x)

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