import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

CLASSES = [0, 1, 2, 3, 4]
COLORMAP = [
    [0, 0, 102],   # background
    [0, 255, 0],     # green
    [255, 0, 0],     # red
    [0, 0, 255],     # blue
    [255, 255, 0],   # yellow

]

def get_transforms(image=None, mask=None, train=False, test=False, base_size=256, multi_scale=False):
    if multi_scale == True:
        min_size = 256
        max_size = 481
        base_size = np.random.randint(low=min_size, high=max_size)
        
    if train == False:
        Transform = albu.Compose([
            albu.Resize(width=base_size, height=base_size, always_apply=True, interpolation=cv2.INTER_NEAREST, p=1),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2(),
            ])
    else:
        Transform = albu.Compose([
            albu.Resize(width=base_size, height=base_size, always_apply=True, interpolation=cv2.INTER_NEAREST, p=1),

            albu.Superpixels(p_replace=0.1, n_segments=128, interpolation=cv2.INTER_NEAREST, p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.VerticalFlip(p=0.5),
            albu.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2(),
            ])
    if test == False:
        sample = Transform(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
        return image, mask
    else:
        sample = Transform(image=image)
        image = sample['image']
        return image
    
class Gleason(Dataset):
    """Gleason Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    """CLASSES = ['background', 'Begin', 'Gleason score 3', 'Gleason score 4', 'Gleason score 5']"""
    """CLASSES = ['0', '1', '3', '4', '5']"""

    def __init__(self, images_dir, masks_dir=None, tranforms=False, train=False, test=False, base_size=256, multi_scale=False):
        self.img_list = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.img_list]
        if masks_dir is not None:
            self.mask_list = [item.replace('.jpg', '_classimg_nonconvex.png') for item in self.img_list]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_list]

        # convert str names to class values on masks
        self.class_values = [0, 1, 3, 4, 5]
        self.get_tranforms = tranforms
        self.train = train
        self.test = test
        self.size = base_size
        self.multi_scale = multi_scale

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(self.images_fps[i])
        if self.test:
            image = get_transforms(image=image, train=False, test=True,  base_size=self.size, multi_scale=self.multi_scale)
            return image

        mask = cv2.imread(self.masks_fps[i], 0)
        # apply tranfrom
        if self.get_tranforms == True:
            image, mask = get_transforms(image=image, mask=mask, train=self.train, test=self.test, base_size=self.size, multi_scale=self.multi_scale)
        return image, mask

    def __len__(self):
        return len(self.img_list)