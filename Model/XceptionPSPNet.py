import torch.nn as nn
from torch.nn import functional as F
from Model.PPM import PPM
from Model.Xception65 import AlignedXception

class XceptionPSPNet(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.2, n_classes=5, zoom_factor=16, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(XceptionPSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert n_classes > 1
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        xception = AlignedXception(output_stride=zoom_factor, BatchNorm=nn.BatchNorm2d, pretrained=True)
        self.entry =  nn.Sequential(
            xception.conv1, xception.bn1, xception.relu,
            xception.conv2, xception.bn2, xception.relu
        )
        self.block1 = xception.block1
        self.relublock1 = xception.relu
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
            xception.conv3, xception.bn3, xception.relu,
            xception.conv4, xception.bn4, xception.relu,
            xception.conv5, xception.bn5, xception.relu
        )

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, n_classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        _, _, h, w = x.size()
        x = self.entry(x)
        x = self.block1(x)
        x = self.relublock1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x_tmp = self.block20(x)
        x = self.block21(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
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