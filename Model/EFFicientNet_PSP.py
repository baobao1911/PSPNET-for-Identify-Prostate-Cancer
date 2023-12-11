import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.Backbone.EfficientNet import *
from Model.Module.HDC import *
from Model.Module.PPM import *
from Model.Module.Gau import *

class EFFicientNet_PSP(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), rates=[1, 2, 5, 1 ,2, 5], dropout=0.25, classes=6, zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(EFFicientNet_PSP, self).__init__()
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion

        backbone_path = r'D:\University\Semantic_Segmentation_for_Prostate_Cancer_Detection\Semantic_Segmentation_for_Prostate_Cancer_Detection\Utils\efficientnet_b3.pth'
        model = efficientNet(pretrained=True, weight='b3', weight_path=backbone_path)
        self.stem = model.stem
        self.blocks = model.blocks
        i = 0
        for block in self.blocks:
            for n, m in block.named_modules():
                if '.0' in n and 'depthwise_conv' in n:
                    if m.stride == (2, 2):
                        i +=1
                        if i > 1:
                            m.stride = (1, 1)
        self.head = model.head

        fea_dim = 2048
        self.ppm = PPM_custom(fea_dim, int(fea_dim/len(bins)), bins, rates)
        fea_dim = int(fea_dim/len(bins))
        self.fc = nn.Sequential(
               nn.Conv2d(fea_dim, 256, kernel_size=3, padding=1, bias=False),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               nn.Dropout2d(p=dropout),
               nn.Conv2d(256, classes, kernel_size=1)
           )
        
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        _, _, h, w = x.size()

        x = self.stem(x)
        x_tmp = self.blocks(x)
        x = self.head(x_tmp)
            
        x = self.ppm(x)
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