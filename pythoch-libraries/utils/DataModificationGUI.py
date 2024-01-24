"""

Custom Autoencoders 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import numpy as np
import cv2
import pandas as pd
from VideoAnnotationHelper import createFolder
import random
import torch
from torchvision.io import read_image, ImageReadMode

device = ("cuda" if torch.cuda.is_available() else "cpu")

def click_event(event, x, y, flags, params): 
    
    if event == cv2.EVENT_LBUTTONDOWN: 
        
        y_crop_size = params[3]
        x_crop_size = params[4]
        
        x1 = x - int(x_crop_size/2) 
        y1 = y - int(y_crop_size/2) 
        x2 = x + int(x_crop_size/2) 
        y2 = y + int(y_crop_size/2)
        
        r = random.random()
        g = random.random()
        b = random.random()
        
        cv2.rectangle(params[0], (x1,y1), (x2,y2), (r,g,b), 1)
        cv2.imshow('image', params[0]) 
        
    if event==cv2.EVENT_RBUTTONDOWN: 
        
        y_crop_size = params[3]
        x_crop_size = params[4]
        
        img2 = params[1][y - int(y_crop_size/2) : y + int(y_crop_size/2), x - int(x_crop_size/2) : x + int(x_crop_size/2)]
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        
        msk2 = params[2][y - int(y_crop_size/2) : y + int(y_crop_size/2), x - int(x_crop_size/2) : x + int(x_crop_size/2)]
        
        cv2.imwrite(params[5] + "/" +  params[7] , img2)
        cv2.imwrite(params[6] + "/" +  params[7] , msk2)
        
        print(params[7], " image saved!")


def CropImageGUI(model_name = "model_cyc_1_ep_1000_bs_32_lr_0.0001_20231031_T191150.pt",
                parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena",
                data_folder = "train_data",
                sub_folder = "EL_2",
                img_dir = "images",
                mask_dir = "masks",
                csv_filename = "original.csv",
                x_crop_size = 415,
                y_crop_size = 256):
    
    model = torch.jit.load(parent_dir + "/models/shaft_segmentation_models/" + model_name)
    df = pd.read_csv(parent_dir + "/" + data_folder + "/" + sub_folder +"/" + csv_filename)
    
    images_path = []
    masks_path = []
    imgMask_names = [] 

    for name in df["Images"]: 
        images_path.append(parent_dir + "/" + data_folder + "/" + sub_folder + "/" + img_dir + "/" + name)
        masks_path.append(parent_dir + "/" + data_folder + "/" + sub_folder + "/" + mask_dir + "/" + name)
        imgMask_names.append(name)
        
       
    new_image_folder = parent_dir + "/" + data_folder + "/" + sub_folder + "/" + "images_outliers_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    new_mask_fodler = parent_dir + "/" + data_folder + "/" + sub_folder + "/" + "masks_outliers_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    
    createFolder(new_image_folder)
    createFolder(new_mask_fodler)
    
    for n in range(len(imgMask_names)):
        
        
        rgb_input_tensor = read_image(images_path[n],mode=ImageReadMode.RGB)
        _, height, width = rgb_input_tensor.size() 
        input_tensor = rgb_input_tensor.type(torch.FloatTensor)
        input_tensor = input_tensor / 255
        input_tensor = input_tensor.to(device)
        input_tensor = torch.reshape(input_tensor,(1,3,height,width))
        
        # For cropping
        img = np.transpose(rgb_input_tensor, (1,2,0)).type(torch.FloatTensor).numpy()
        
        ture_label = read_image(masks_path[n])
        ture_label = ture_label.type(torch.FloatTensor)
        
        # For cropping 
        msk = np.transpose(ture_label, (1,2,0)).numpy()
        
        
        with torch.no_grad(): 
            pred_label = model.forward(input_tensor)
                                       
        pred_label = np.transpose(pred_label[0].cpu(),(1,2,0)).numpy()
                                       
        masked_b = np.zeros([1280,720], dtype=np.float32)
        merged_img = cv2.merge((masked_b, pred_label, masked_b))
                                       
        overlap_img = cv2.addWeighted(img/255, 0.5, merged_img, 0.5, 0)
        overlap_img = cv2.cvtColor(overlap_img,cv2.COLOR_RGB2BGR)
        
        params = [overlap_img, img, msk, y_crop_size, x_crop_size, new_image_folder, new_mask_fodler, imgMask_names[n]]
        
        cv2.imshow('image', overlap_img)
        cv2.setMouseCallback('image', click_event, params) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__": 
    
    model_name = "model_cyc_1_ep_1000_bs_32_lr_0.0001_20231031_T191150.pt"
    parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena"
    data_folder = "test_data"
    sub_folder = "LiveTest"
    img_dir = "images"
    mask_dir = "masks"
    csv_filename = "cropped_256x256.csv"
    x_crop_size = 256
    y_crop_size = 256
    
    CropImageGUI(model_name = model_name,
                parent_dir = parent_dir,
                data_folder = data_folder,
                sub_folder = sub_folder,
                img_dir = img_dir,
                mask_dir = mask_dir,
                csv_filename = csv_filename,
                x_crop_size = x_crop_size,
                y_crop_size = y_crop_size) 
        