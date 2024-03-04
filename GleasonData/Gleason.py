import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2

CLASSES = [0, 1, 2, 3, 4]
COLORMAP = [
    [0, 0, 102],   # background
    [0, 255, 0],     # green
    [0, 0, 255],     # blue
    [255, 255, 0],   # yellow
    [255, 0, 0],     # red
]
class CFG:
    train_bs      = 64
    valid_bs      = train_bs*2
    img_size      = [256, 256]
    epochs        = 100
    n_workers     = 2

data_transforms = {
    "train": albu.Compose([
        albu.Resize(*CFG.img_size, interpolation=cv2.INTER_LINEAR, p=1, always_apply=True),
        albu.HorizontalFlip(p=0.5),
#         albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        albu.OneOf([
            albu.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
# #             albu.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        ToTensorV2(),
        ], p=1.0),
    
    "valid": albu.Compose([
        albu.Resize(*CFG.img_size, interpolation=cv2.INTER_LINEAR, p=1, always_apply=True),
        #ToTensorV2(),
        ], p=1.0)
}
class Gleason():
    def __init__(self, images_dir, masks_dir=None, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        self.img_list = os.listdir(images_dir)
        self.img_list.sort()
        if masks_dir is not None:
            self.mask_list = os.listdir(masks_dir)
            self.mask_list.sort()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        print(self.img_list[i])
        print(self.mask_list[i])
        img = cv2.imread(f'{self.images_dir}/{self.img_list[i]}', 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        img = np.array(img, np.float32)     
        img /=255.0
        if self.masks_dir is not None:
            msk = cv2.imread(f'{self.masks_dir}/{self.mask_list[i]}', cv2.IMREAD_GRAYSCALE) 
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            return img, msk
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            return img
        
def prepare_loaders(train_img=None, train_mask=None, 
                    valid_img=None, valid_mask=None,
                    img_size=[256, 256], n_workers=2):
    CFG.img_size = img_size
    CFG.n_workers = n_workers
    train_dataset = Gleason(train_img, train_mask, transforms=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, 
                            num_workers=CFG.n_workers, shuffle=True, pin_memory=True, drop_last=True)

    valid_dataset = Gleason(valid_img, valid_mask, transforms=data_transforms['valid'])
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, 
                            num_workers=CFG.n_workers, shuffle=False, pin_memory=True)
    return train_loader, valid_loader
