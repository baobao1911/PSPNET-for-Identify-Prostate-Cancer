import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rates):
        super(HybridDilatedConv, self).__init__()

        effective_dilation = [rate * (kernel_size - 1) for rate in rates]
        p = [int((effective_dilation[i] - 1) / 2) for i in range(len(rates))]

        self.dilated_conv = []
        for i, rate in enumerate(rates):
            self.dilated_conv.append(nn.Conv2d(in_channels, out_channels // len(rates), kernel_size, padding=p[i]+1, dilation=rate, bias=False))
        self.dilated_conv = nn.ModuleList(self.dilated_conv)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = []
        for hdc in self.dilated_conv:
            out.append(hdc(x))
        # for i in out:
        #     print(i.size())
        combined_features = torch.cat(out, 1)
        output = self.bn(combined_features)
        output = self.relu(output)
        return output