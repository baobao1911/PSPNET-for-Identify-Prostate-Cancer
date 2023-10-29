import os
import cv2
import numpy as np

from torch.utils.data import Dataset

CLASSES = [0, 1, 2, 3, 4]
COLORMAP = [
    [0, 0, 102],   # background
    [0, 255, 0],     # green
    [255, 0, 0],     # red
    [0, 0, 255],     # blue
    [255, 255, 0],   # yellow

]
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

    """CLASSES = ['background', 'Begin', 'Gleason score 5', 'Gleason score 3', 'Gleason score 4']"""
    """CLASSES = ['0', '1', '2', '3', '4']"""

    def __init__(self, images_dir, masks_dir, augmentation=None,
                 preprocessing=None, test=False):
        self.img_list = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.img_list]

        self.mask_list = [item.replace('.jpg', '_classimg_nonconvex.png') for item in self.img_list]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_list]

        # convert str names to class values on masks
        self.class_values = [0, 1, 2, 3, 4, 5]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.test = test

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.test:
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
            return image

        mask = cv2.imread(self.masks_fps[i], 0)

        # # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation != None:
            samples = self.augmentation(image=image, mask=mask)
            image, mask = samples['image'], samples['mask']

        # apply tranfrom
        if self.preprocessing != None:
            samples = self.preprocessing(image=image, mask=mask)
            image, mask = samples['image'], samples['mask']
        return image, mask

    def __len__(self):
        return len(self.img_list)