import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from GleasonData.Gleason import *
from Model.PSPNet_Custom import PSPNet_Custom
from Model.PSPNet import PSPNet
from Utils.utils import *

def build_testing(model, device, img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predict_mask = np.zeros_like(image)
    sliding_window = 5
    model.eval()
    # Calculate the new dimensions for the cropped image
    new_height = np.shape(image)[0] // sliding_window
    new_width = np.shape(image)[1] // sliding_window
    for h in range(0, np.shape(image)[0] - new_height+1, new_height):
      for w in range(0, np.shape(image)[1] - new_width+1, new_width):
        sub_image = image[h:h+new_height, w:w+new_width]
        sub_image = get_transforms(image=sub_image, train=False, test=True,  base_size=284, multi_scale=False)

        sub_mask_predict = model(sub_image.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu().numpy()
        predict_mask[h:h+new_height, w:w+new_width] = sub_mask_predict

    return image, predict_mask

def show_mask(image, predict_mask):
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(predict_mask)
    plt.show()
    
if __name__ == "__main__":
    model_path = '/home/bao/Project/PSPNet_Custom10.pth'
    image_path = 'Data/Test_imgs/slide001_core015.jpg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device used: {device}')
    model = PSPNet_Custom(pretrained=True, Backbone_path='/home/bao/Downloads/resnet101-cd907fc2.pth')
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)
    image, preditc = build_testing(model, device, image_path)
    show_mask(image, preditc)

