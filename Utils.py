import torch
from GleasonData.Gleason import Gleason
import DataTransforms as Transfroms

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
    return area_intersection, area_union, area_target

def get_dataset(img_path, mask_path, augmentation=None, preprocessing=None, test=False):
    dataset = Gleason(img_path, mask_path, augmentation=augmentation, 
                      preprocessing=preprocessing, test=test)
    return dataset

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


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

def get_transforms(train):
    base_size = 1000
    crop_size = 768

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(Transfroms.RandomResize(min_size, max_size))
    if train:
        transforms.append(Transfroms.RandomHorizontalFlip(0.5))
        transforms.append(Transfroms.ColorJitter(0.5, 0.5, 0.5, 0.5))
        transforms.append(Transfroms.RandomVerticalFlip(0.5))
        transforms.append(Transfroms.RandomCrop(crop_size))
    transforms.append(Transfroms.ToTensor())
    transforms.append(Transfroms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return Transfroms.Compose(transforms)