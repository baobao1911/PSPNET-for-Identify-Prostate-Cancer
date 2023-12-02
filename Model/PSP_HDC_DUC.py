import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Backbone.Resnet101 import *
from Model.Module.HDC import *
from Model.Module.PPM import *

class DUC(nn.Module):
    def __init__(self, classes, batch_size, zoom_factor):
        super(DUC, self).__init__()
        self.classes = classes
        self.b = batch_size
        self.r = zoom_factor
    def forward(self, x):
        new_x = torch.zeros(self.b, self.classes, 304, 304)
        for i in range(0, 32):
            for j in range(0, 32):
                tmp = x[:,:, i:i+1, j:j+1]
                tmp = tmp.view(self.b, self.classes, self.r**2, 1, 1)
                tmp = tmp.view(self.b, self.classes, self.r, self.r)
                new_x[:, :, i*self.r:i*self.r+self.r, j*self.r:j*self.r+self.r] = tmp
        return new_x


class PSP_HDC_DUC(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), rates=[1, 2, 5, 1, 2, 5], dropout=0.3, classes=6, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSP_HDC_DUC, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.classes = classes

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
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.hdc = HybridDilatedConv(fea_dim, kernel_size=3, rates=rates)

        fea_dim = fea_dim*2 + int(fea_dim//4)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes*zoom_factor**2, kernel_size=1)
        )
        self.duc = DUC(classes, 4, zoom_factor)

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
    
    def forward(self, x, y=None):
        b, c, h, w = x.size()

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        x_ppm = self.ppm(x)
        x_hdc = self.hdc(x)
        x = torch.cat([x_hdc, x_ppm], dim=1)
        x = self.cls(x)

        x = self.duc(x.clone())
        #x = self.softmax(x)
        _, _, _, w4 = x.size()
        if w4 != w:
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