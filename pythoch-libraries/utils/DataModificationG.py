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
import glob
import os
from copy import deepcopy
from torchvision.io import read_image, ImageReadMode

def click_event(event, x, y, flags, params): 
    
    if event == cv2.EVENT_LBUTTONDOWN: 
        
        y_crop_size = params[1]
        x_crop_size = params[2]
        
        x1 = x - int(x_crop_size/2) 
        y1 = y - int(y_crop_size/2)
        x2 = x + int(x_crop_size/2) 
        y2 = y + int(y_crop_size/2)
        
        r = int(255 * random.random())
        g = int(255 * random.random())
        b = int(255 * random.random())
        
        cv2.rectangle(params[0], (x1,y1), (x2,y2), (r,g,b), 1)
        cv2.imshow('image', params[0]) 
        
    if event==cv2.EVENT_RBUTTONDOWN: 
        
        y_crop_size = params[1]
        x_crop_size = params[2]
        
        img2 = params[5][y - int(y_crop_size/2) : y + int(y_crop_size/2), x - int(x_crop_size/2) : x + int(x_crop_size/2)]
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(params[3] + "/" +  params[4] , img2)
        
        print(params[4], " image saved!")

        

def CropImageGUI(parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena",
                data_folder = "train_data",
                sub_folder = "EL_2",
                x_crop_size = 256, 
                y_crop_size = 256):

    images_path = glob.glob(parent_dir + "/" + data_folder + "/" + sub_folder + "/*.png")
    
    new_img_folder = parent_dir + "/" + data_folder + "/images_" + str(y_crop_size)  + "x" + str(x_crop_size)
    
    createFolder(new_img_folder)
    
    for n in range((len(images_path))):
        rgb_input_tensor = read_image(images_path[n],mode=ImageReadMode.RGB)
        img_name = os.path.basename(images_path[n])
        img = np.transpose(rgb_input_tensor, (1,2,0)).numpy()
        img_copy = deepcopy(img)
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        params = [img, y_crop_size, x_crop_size, new_img_folder, img_name, img_copy]
        
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event, params)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__": 
    
    parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/"
    data_folder = "test/HN2/"
    sub_folder = "images"
    x_crop_size = 256 
    y_crop_size = 256
    
    CropImageGUI(parent_dir = parent_dir,
                data_folder = data_folder, 
                sub_folder = sub_folder, 
                x_crop_size = x_crop_size, 
                y_crop_size = y_crop_size)
    
    