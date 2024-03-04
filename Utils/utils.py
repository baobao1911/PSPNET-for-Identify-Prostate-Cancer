import torch
import numpy as np
import torch.optim as optim


    
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target, area_output

def IoU(area_intersection, area_union, smooth= 1e-10):
    iou = area_intersection / (area_union + smooth)
    return np.mean(iou)

def Dice(area_intersection, area_target, area_output, smooth=1e-10):
    dice = (2*area_intersection)/(area_output + area_target + smooth)
    return np.mean(dice)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fetch_scheduler(optimizer, scheduler, epochs, max_iter):
    if scheduler == 'PolicyLR':
        try:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
                                    lambda x: (1 - x / (max_iter * epochs)) ** 0.9)
        except ValueError:
            raise ValueError("check number train_loader")
    elif scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    else:
        return None
    return scheduler