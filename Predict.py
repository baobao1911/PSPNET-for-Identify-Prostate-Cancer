import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from GleasonData.Gleason import *
from Model.PSPNet_Custom import PSPNet_Custom
from Model.PSPNet import PSPNet
from Utils.utils import *

def build_testing(model, device, img_path, base_size=384):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sliding_window = 5
    # Calculate the new dimensions for the cropped image
    new_height = np.shape(image)[0] // sliding_window
    new_width = np.shape(image)[1] // sliding_window
    model.eval()
    predict_list = []
    image_list = []
    position = []
    for h in range(0, np.shape(image)[0] - new_height+1, new_height):
        for w in range(0, np.shape(image)[1] - new_width+1, new_width):
            sub_image = image[h:h+new_height, w:w+new_width]
            image_list.append(sub_image)
            sub_image = get_transforms(image=sub_image, train=False, test=True,  base_size=base_size, multi_scale=False)
            sub_mask_predict = model(sub_image.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu().numpy()
            sub_mask_predict[sub_mask_predict == 2] = 0
            predict_list.append(sub_mask_predict)
            position.append([h, w])

    return image_list, predict_list, position

def show_mask(image, predict_mask):
    custom_cmap = ListedColormap(['gray', 'green', 'black', 'blue', 'yellow', 'red'])


    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout(pad=3)
    print(np.unique(predict_mask))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(predict_mask, cmap=custom_cmap, vmin=0, vmax=5)
    plt.show()
    
if __name__ == "__main__":
    model_path = 'Training_result/ModelSave/PSPNet_Custom10.pth'
    image_path = 'Data/Test_imgs/slide001_core015.jpg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device used: {device}')
    model = PSPNet_Custom(pretrained=True, Backbone_path=r'D:\University\Semantic_Segmentation_for_Prostate_Cancer_Detection\Semantic_Segmentation_for_Prostate_Cancer_Detection\Utils\resnext50_32x4d-1a0047aa.pth')
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)
    image, preditc = build_testing(model, device, image_path)
    show_mask(image, preditc)

