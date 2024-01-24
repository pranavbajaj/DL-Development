"""
Custom Autoencoders 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import csv
import os 
import glob
import torch
import numpy as np
import cv2
import random
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import random
import math 
import pandas as pd
from sklearn.linear_model import LinearRegression as lr 
from VideoAnnotationHelper import createFolder, LineFitOnMask, LineMaskfloat32
device = ("cuda" if torch.cuda.is_available() else "cpu")

def dataModification(parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/train_data/EL_2/",
                    txt_filename="groundtruth_2point.txt",
                    y_crop_size = 256,
                    x_crop_size = 256,
                    height = 1280, 
                    width = 720, 
                    ifSaveImages = True,
                    ifSaveCSV = True):
    
    
#     if y_crop_size % 8 != 0 or x_crop_size % 8 != 0: 
#         raise Exception("Crop size should be multiple of 8!")
    
    cropped_images_dir = "images_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masks_dir = "masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_hub_masks_dir = "hub_masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_tip_masks_dir = "tip_masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masksLined_dir = "masksLined_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    csv_file_name = parent_dir + "/cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + ".csv"
    
    createFolder(parent_dir + cropped_images_dir)
    createFolder(parent_dir + cropped_masks_dir)
    createFolder(parent_dir + cropped_masksLined_dir)
    createFolder(parent_dir + cropped_hub_masks_dir)
    createFolder(parent_dir + cropped_tip_masks_dir)
    
    file_path = os.path.join(parent_dir,txt_filename)
    f = open(file_path, "r")
    data = csv.reader(f, delimiter='\t')
    
    isHub = 1
    
    images_name = []
    hx_crop_loc = []
    hy_crop_loc = []
    tx_crop_loc = []
    ty_crop_loc = []
    hubs = []
    
    for row in data: 
        i_name = row[0]
        images_name.append(i_name)
        print(i_name)
    
    
        if len(row) == 5: 
            isHub = 1
            hubs.append(isHub)
            hub_x = int(row[1])
            hub_y = int(row[2])
            tip_x = int(row[3])
            tip_y = int(row[4])
        else: 
            isHub = 0
            hubs.append(isHub)
            hx_crop_loc.append(-1)
            hy_crop_loc.append(-1)
            tx_crop_loc.append(-1)
            ty_crop_loc.append(-1)
        
        y = []
        x = []

        if isHub:
            y_mean = random.randint(hub_y - int(y_crop_size/2) + 10, hub_y + int(y_crop_size/2) - 10)
            x_mean = random.randint(hub_x - int(x_crop_size/2) + 10, hub_x + int(x_crop_size/2) - 10) 
        else: 
            msk_path = parent_dir + "masks/" + i_name 
            msk = read_image(msk_path)
            msk = np.transpose(msk.numpy(), (1,2,0))
            for i in range(len(msk)): 
                for j in range(len(msk[0])):

                    if msk[i][j][0] > 0:
                        y.append(i)
                        x.append(j)

            y_mean = int(sum(y)/len(y))
            x_mean = int(sum(x)/len(x))

            
        y_min = y_mean - int(y_crop_size/2)
        x_min = x_mean - int(x_crop_size/2)
        y_max = y_mean + int(y_crop_size/2)
        x_max = x_mean + int(x_crop_size/2)

        if y_min < 0: 
            y_min = 0 
            y_max = y_crop_size

        elif y_max > (height):
            y_max = height 
            y_min = y_max - y_crop_size

        if x_min < 0: 
            x_min = 0 
            x_max = x_crop_size

        elif x_max > (width):
            x_max = width 
            x_min = x_max - x_crop_size

        
        
        if isHub:

            x1 = hub_x - x_min 
            y1 = hub_y - y_min 
            hx_crop_loc.append(x1)
            hy_crop_loc.append(y1)

            x2 = tip_x - x_min
            y2 = tip_y - y_min 

            if (x2 < x_crop_size) and (x2 >= 0) and (y2 < y_crop_size) and (y2 >= 0):
                ty_crop_loc.append(y2)
                tx_crop_loc.append(x2)
            else: 

                if (x2 > x1):
                    x2_1 = x_crop_size - 1 
                else: 
                    x2_1 = 0
                
                if x2 != x1:
                    y2_1 = y1 + int((y2 - y1) * (x2_1 - x1) / (x2 - x1))
                else: 
                    y2_1 = 3000


                if (y2 > y1):
                    y2_2 = y_crop_size - 1
                else: 
                    y2_2 = 0
                    
                if (y2 != y1):
                    x2_2 = x1 + int((x2 - x1) * (y2_2 - y1) / (y2 - y1)) 
                else: 
                    x2_2 = 3000 
               

                if  (y2_1 < y_crop_size) and (y2_1 >= 0):
                    ty_crop_loc.append(y2_1)
                    tx_crop_loc.append(x2_1)
                else: 
                    ty_crop_loc.append(y2_2)
                    tx_crop_loc.append(x2_2)
            
        
        
        if ifSaveImages: 
            img_path = parent_dir + "images/" + i_name 
            img = read_image(img_path)
            img = np.transpose(img.numpy(), (1,2,0))
            
            cropped_lined_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)    
            cropped_hub_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)
            cropped_tip_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)
            
            if isHub: 
                msk_path = parent_dir + "masks/" + i_name 
                msk = read_image(msk_path)
                msk = np.transpose(msk.numpy(), (1,2,0))
                
                hx = hx_crop_loc[len(hx_crop_loc)-1]
                hy = hy_crop_loc[len(hy_crop_loc)-1]
                tx = tx_crop_loc[len(tx_crop_loc)-1]
                ty = ty_crop_loc[len(ty_crop_loc)-1]
                
                cropped_hub_mask[hy][hx] = 1.0
                cropped_tip_mask[ty][tx] = 1.0
                
                if (hx != tx):
                    X = np.array([[hx],[tx]])
                    y = np.array([hy,ty])
                    reg = lr().fit(X,y)
                    X_range = np.linspace(0, x_crop_size, 1000)
                    X_range = X_range.reshape(-1,1)
                    y_range = reg.predict(X_range)
                    for i in range(len(y_range)): 
                        if int(y_range[i]) >= 0 and int(y_range[i]) < y_crop_size:
                            if int(X_range[i][0]) >= 0 and int(X_range[i][0]) < x_crop_size: 
                                cropped_lined_mask[int(y_range[i])][int(X_range[i][0])] = 1.0
                                
                else:
                    y_range = [i for i in range(0,y_crop_size,1)]
                    print(y_range)
                    for i in range(len(y_range)): 
                        cropped_lined_mask[int(y_range[i])][int(hx)] = 1.0

            
            cropped_msk = msk[y_min:y_max, x_min:x_max]
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
            
            save_path1 = parent_dir + cropped_masks_dir + "/" + i_name
            cv2.imwrite(save_path1, cropped_msk)

            save_path2 = parent_dir + cropped_images_dir + "/" + i_name
            cv2.imwrite(save_path2, cropped_img)
            
            save_path3 = parent_dir + cropped_masksLined_dir + "/" + i_name
            cv2.imwrite(save_path3, cropped_lined_mask)
            
            save_path4 = parent_dir + cropped_hub_masks_dir + "/" + i_name
            cv2.imwrite(save_path4, cropped_hub_mask)
            
            save_path5 = parent_dir + cropped_tip_masks_dir + "/" + i_name
            cv2.imwrite(save_path5, cropped_tip_mask)
            
    
    if ifSaveCSV: 
        df_images_name = pd.DataFrame(images_name, columns=['Images'])
        df_hubs = pd.DataFrame(hubs, columns=['Hub Present'])
        df_hx_crop_loc = pd.DataFrame(hx_crop_loc, columns=['Hub X'])
        df_hy_crop_loc = pd.DataFrame(hy_crop_loc, columns=['Hub Y'])
        df_tx_crop_loc = pd.DataFrame(tx_crop_loc, columns=['Tip X'])
        df_ty_crop_loc = pd.DataFrame(ty_crop_loc, columns=['Tip Y'])
    
        df2 = pd.concat([df_images_name, df_hx_crop_loc, df_hy_crop_loc, df_tx_crop_loc, df_ty_crop_loc, df_hubs], axis = 1, sort = False)
        df2.to_csv(csv_file_name, index = True)
        
        
        
def dataModificationWithoutSegmentation(parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/train_data/EL_2/",
                    txt_filename="groundtruth_2point.txt",
                    y_crop_size = 256,
                    x_crop_size = 256,
                    height = 1280, 
                    width = 720, 
                    ifSaveImages = True,
                    ifSaveCSV = True):
    
    

    
    cropped_images_dir = "images_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    
    cropped_masksLined_dir = "masksLined_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    csv_file_name = parent_dir + "/cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + ".csv"
    
    createFolder(parent_dir + cropped_images_dir)
    
    createFolder(parent_dir + cropped_masksLined_dir)
    
    file_path = os.path.join(parent_dir,txt_filename)
    f = open(file_path, "r")
    data = csv.reader(f, delimiter='\t')
    
    isHub = 1
    
    images_name = []
    hx_crop_loc = []
    hy_crop_loc = []
    tx_crop_loc = []
    ty_crop_loc = []
    hubs = []
    
    for row in data: 
        if len(row) == 5: 
            i_name = row[0]
            print(i_name)
            images_name.append(i_name)
            isHub = 1
            hubs.append(isHub)
            hub_x = int(row[1])
            hub_y = int(row[2])
            tip_x = int(row[3])
            tip_y = int(row[4])
        
            y = []
            x = []


            y_mean = random.randint(hub_y - int(y_crop_size/2) + 10, hub_y + int(y_crop_size/2) - 10)
            x_mean = random.randint(hub_x - int(x_crop_size/2) + 10, hub_x + int(x_crop_size/2) - 10) 

            y_min = y_mean - int(y_crop_size/2)
            x_min = x_mean - int(x_crop_size/2)
            y_max = y_mean + int(y_crop_size/2)
            x_max = x_mean + int(x_crop_size/2)

            if y_min < 0: 
                y_min = 0 
                y_max = y_crop_size

            elif y_max > (height):
                y_max = height 
                y_min = y_max - y_crop_size

            if x_min < 0: 
                x_min = 0 
                x_max = x_crop_size

            elif x_max > (width):
                x_max = width 
                x_min = x_max - x_crop_size

        
 

            x1 = hub_x - x_min 
            y1 = hub_y - y_min 
            hx_crop_loc.append(x1)
            hy_crop_loc.append(y1)

            x2 = tip_x - x_min
            y2 = tip_y - y_min 

            if (x2 < x_crop_size) and (x2 >= 0) and (y2 < y_crop_size) and (y2 >= 0):
                ty_crop_loc.append(y2)
                tx_crop_loc.append(x2)
            else: 

                if (x2 > x1):
                    x2_1 = x_crop_size - 1 
                else: 
                    x2_1 = 0
                
                if x2 != x1:
                    y2_1 = y1 + int((y2 - y1) * (x2_1 - x1) / (x2 - x1))
                else: 
                    y2_1 = 3000


                if (y2 > y1):
                    y2_2 = y_crop_size - 1
                else: 
                    y2_2 = 0
                    
                if (y2 != y1):
                    x2_2 = x1 + int((x2 - x1) * (y2_2 - y1) / (y2 - y1)) 
                else: 
                    x2_2 = 3000 
               

                if  (y2_1 <= y_crop_size) and (y2_1 >= 0):
                    ty_crop_loc.append(y2_1)
                    tx_crop_loc.append(x2_1)
                else: 
                    ty_crop_loc.append(y2_2)
                    tx_crop_loc.append(x2_2)
            
        
        
            if ifSaveImages: 
                img_path = parent_dir + "/images/" + i_name 
                img = read_image(img_path)
                img = np.transpose(img.numpy(), (1,2,0))


                hx = hx_crop_loc[len(hx_crop_loc)-1]
                hy = hy_crop_loc[len(hy_crop_loc)-1]
                tx = tx_crop_loc[len(tx_crop_loc)-1]
                ty = ty_crop_loc[len(ty_crop_loc)-1]

                cropped_lined_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)

                if (hx != tx):
                    X = np.array([[hx],[tx]])
                    y = np.array([hy,ty])
                    reg = lr().fit(X,y)
                    X_range = np.linspace(0, x_crop_size, 1000)
                    X_range = X_range.reshape(-1,1)
                    y_range = reg.predict(X_range)
                    for i in range(len(y_range)): 
                        if int(y_range[i]) >= 0 and int(y_range[i]) < y_crop_size:
                            if int(X_range[i][0]) >= 0 and int(X_range[i][0]) < x_crop_size: 
                                cropped_lined_mask[int(y_range[i])][int(X_range[i][0])] = 1.0

                else:
                    y_range = [i for i in range(0,y_crop_size,1)]
                    print(y_range)
                    for i in range(len(y_range)): 
                        cropped_lined_mask[int(y_range[i])][int(hx)] = 1.0

                cropped_img = img[y_min:y_max, x_min:x_max]
                cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

                save_path2 = parent_dir + cropped_images_dir + "/" + i_name
                cv2.imwrite(save_path2, cropped_img)

                save_path3 = parent_dir + cropped_masksLined_dir + "/" + i_name
                cv2.imwrite(save_path3, cropped_lined_mask)
    
    if ifSaveCSV: 
        df_images_name = pd.DataFrame(images_name, columns=['Images'])
        df_hubs = pd.DataFrame(hubs, columns=['Hub Present'])
        df_hx_crop_loc = pd.DataFrame(hx_crop_loc, columns=['Hub X'])
        df_hy_crop_loc = pd.DataFrame(hy_crop_loc, columns=['Hub Y'])
        df_tx_crop_loc = pd.DataFrame(tx_crop_loc, columns=['Tip X'])
        df_ty_crop_loc = pd.DataFrame(ty_crop_loc, columns=['Tip Y'])
    
        df2 = pd.concat([df_images_name, df_hx_crop_loc, df_hy_crop_loc, df_tx_crop_loc, df_ty_crop_loc, df_hubs], axis = 1, sort = False)
        df2.to_csv(csv_file_name, index = True)        

def dataModificationOriginal(parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/train_data/EL_2/",
                    txt_filename="groundtruth_2point.txt",
                    y_crop_size = 256,
                    x_crop_size = 256,
                    height = 1280, 
                    width = 720, 
                    ifSaveImages = True,
                    ifSaveCSV = True):
    
#     if y_crop_size % 8 != 0 or x_crop_size % 8 != 0: 
#         raise Exception("Crop size should be an even number!")

    cropped_images_dir = "images_cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + "_original"
    cropped_masks_dir = "masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + "_original"
    cropped_masksLined_dir = "masksLined_cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + "_original"
    cropped_hub_masks_dir = "hub_masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + "_original"
    cropped_tip_masks_dir = "tip_masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + "_original"
    
    csv_file_name = parent_dir + "/cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + "_original" + ".csv"
    
    createFolder(parent_dir + cropped_images_dir)
    createFolder(parent_dir + cropped_masks_dir)
    createFolder(parent_dir + cropped_masksLined_dir)
    createFolder(parent_dir + cropped_hub_masks_dir)
    createFolder( parent_dir + cropped_tip_masks_dir)
                 
    file_path = os.path.join(parent_dir,txt_filename)
    f = open(file_path, "r")
    data = csv.reader(f, delimiter='\t')
    
    isHub = 1
    
    images_name = []
    hx_crop_loc = []
    hy_crop_loc = []
    tx_crop_loc = []
    ty_crop_loc = []
    hubs = []
    
    for row in data: 
        i_name = row[0]
        print(i_name)
    
    
        if len(row) == 5: 
            isHub = 1
            hubs.append(isHub)
            hub_x = int(row[1])
            hub_y = int(row[2])
            tip_x = int(row[3])
            tip_y = int(row[4])
    
        else: 
            isHub = 0
            hubs.append(isHub)
            hx_crop_loc.append(-1)
            hy_crop_loc.append(-1)
            tx_crop_loc.append(-1)
            ty_crop_loc.append(-1)
        
        y = []
        x = []
        
        img_path = parent_dir + "images/" + i_name 
        img = read_image(img_path)
        img = np.transpose(img.numpy(), (1,2,0))

        msk_path = parent_dir + "masks/" + i_name 
        msk = read_image(msk_path)
        msk = np.transpose(msk.numpy(), (1,2,0))
        
        
        for i in range(len(msk)): 
            for j in range(len(msk[0])): 

                if msk[i][j][0] > 0: 
                    y.append(i)
                    x.append(j)

        if len(y) == 0:
            continue 

        images_name.append(i_name)
        if len(y) == 0: 
            y_mean = 500
            x_mean = 500 
        else:     
            y_mean = int(sum(y)/len(y))
            x_mean = int(sum(x)/len(x))

        if isHub:
            if (y_mean > hub_y):
                y_mean = random.randint(hub_y ,y_mean)
            else: 
                y_mean = random.randint(y_mean, hub_y)
                
            if (x_mean > hub_x):
                x_mean = random.randint(hub_x, x_mean)
            else: 
                x_mean = random.randint(x_mean, hub_x)
        
        
        y_min = y_mean - int(y_crop_size/2)
        x_min = x_mean - int(x_crop_size/2)
        y_max = y_mean + int(y_crop_size/2)
        x_max = x_mean + int(x_crop_size/2)
        
        if y_min < 0: 
            y_min = 0 
            y_max = y_crop_size 

        elif y_max > (height):
            y_max = height 
            y_min = y_max - y_crop_size

        if x_min < 0: 
            x_min = 0 
            x_max = x_crop_size 

        elif x_max > (width):
            x_max = width 
            x_min = x_max - x_crop_size

        if isHub: 
            x1 = hub_x - x_min 
            y1 = hub_y - y_min 
            hx_crop_loc.append(x1)
            hy_crop_loc.append(y1)

            x2 = tip_x - x_min
            y2 = tip_y - y_min 

            if (x2 < x_crop_size) and (x2 >= 0) and (y2 < y_crop_size) and (y2 >= 0):
                ty_crop_loc.append(y2)
                tx_crop_loc.append(x2)
            else: 

                if (x2 > x1):
                    x2_1 = x_crop_size - 1
                else: 
                    x2_1 = 0
                
                if x2 != x1:
                    y2_1 = y1 + int((y2 - y1) * (x2_1 - x1) / (x2 - x1))
                else: 
                    y2_1 = 3000


                if (y2 > y1):
                    y2_2 = y_crop_size - 1
                else: 
                    y2_2 = 0
                    
                if (y2 != y1):
                    x2_2 = x1 + int((x2 - x1) * (y2_2 - y1) / (y2 - y1)) 
                else: 
                    x2_2 = 3000 


                if  (y2_1 < y_crop_size) and (y2_1 >= 0):
                    ty_crop_loc.append(y2_1)
                    tx_crop_loc.append(x2_1)
                else: 
                    ty_crop_loc.append(y2_2)
                    tx_crop_loc.append(x2_2)
            
        if ifSaveImages: 
            
            cropped_lined_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)
            cropped_hub_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)
            cropped_tip_mask = np.zeros([y_crop_size,x_crop_size], dtype =np.float32)
                
                
            if isHub: 
                hx = hx_crop_loc[len(hx_crop_loc)-1]
                hy = hy_crop_loc[len(hy_crop_loc)-1]
                tx = tx_crop_loc[len(tx_crop_loc)-1]
                ty = ty_crop_loc[len(ty_crop_loc)-1]   
                
                cropped_hub_mask[hy][hx] = 1.0
                cropped_tip_mask[ty][tx] = 1.0
                
                if (hx != tx):
                    X = np.array([[hx],[tx]])
                    y = np.array([hy,ty])
                    reg = lr().fit(X,y)
                    X_range = np.linspace(0, x_crop_size, 1000)
                    X_range = X_range.reshape(-1,1)
                    y_range = reg.predict(X_range)
                    for i in range(len(y_range)): 
                        if int(y_range[i]) >= 0 and int(y_range[i]) < y_crop_size:
                            if int(X_range[i][0]) >= 0 and int(X_range[i][0]) < x_crop_size: 
                                cropped_lined_mask[int(y_range[i])][int(X_range[i][0])] = 1.0
                                
                else:
                    y_range = [i for i in range(0,y_crop_size,1)]
                    print(y_range)
                    for i in range(len(y_range)): 
                        cropped_lined_mask[int(y_range[i])][int(hx)] = 1.0
                        
                        
            cropped_msk = msk[y_min:y_max, x_min:x_max]
            cropped_img = img[y_min:y_max, x_min:x_max]

            cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

            # Save the cropped mask 
            save_path1 = parent_dir + cropped_masks_dir + "/" + i_name
            cv2.imwrite(save_path1, cropped_msk)

            # Save the cropped image
            save_path2 = parent_dir + cropped_images_dir + "/" + i_name
            cv2.imwrite(save_path2, cropped_img)
            
            save_path3 = parent_dir + cropped_masksLined_dir + "/" + i_name
            cv2.imwrite(save_path3, cropped_lined_mask)
            
            save_path4 = parent_dir + cropped_hub_masks_dir + "/" + i_name
            cv2.imwrite(save_path4, cropped_hub_mask)
            
            save_path5 = parent_dir + cropped_tip_masks_dir + "/" + i_name
            cv2.imwrite(save_path5, cropped_tip_mask)
            
            
    if ifSaveCSV: 
        df_images_name = pd.DataFrame(images_name, columns=['Images'])
        df_hubs = pd.DataFrame(hubs, columns=['Hub Present'])
        df_hx_crop_loc = pd.DataFrame(hx_crop_loc, columns=['Hub X'])
        df_hy_crop_loc = pd.DataFrame(hy_crop_loc, columns=['Hub Y'])
        df_tx_crop_loc = pd.DataFrame(tx_crop_loc, columns=['Tip X'])
        df_ty_crop_loc = pd.DataFrame(ty_crop_loc, columns=['Tip Y'])
    
        df2 = pd.concat([df_images_name, df_hx_crop_loc, df_hy_crop_loc, df_tx_crop_loc, df_ty_crop_loc, df_hubs], axis = 1, sort = False)
        df2.to_csv(csv_file_name, index = True)
        
        
def dataModificationNoHub(parent_dir,
                         txt_filename, 
                         y_crop_size, 
                         x_crop_size, 
                         height = 1280,
                         width = 720):
    
    cropped_images_dir = "images_no_hub_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masks_dir = "masks_no_hub_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masksLined_dir = "masksLined_no_hub_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    
    csv_file_name = parent_dir + "/no_hub_cropped_" + str(y_crop_size) + "x" + str(x_crop_size) + ".csv"
    
    createFolder(parent_dir + cropped_images_dir)
    createFolder(parent_dir + cropped_masksLined_dir)
    
    file_path = os.path.join(parent_dir,txt_filename)
    f = open(file_path, "r")
    data = csv.reader(f, delimiter='\t')
    
    images_name = []
    hx_crop_loc = []
    hy_crop_loc = []
    tx_crop_loc = []
    ty_crop_loc = []
    hubs = []

    for row in data: 

        if len(row) != 5: 
            continue 

        else: 

            i_name = row[0]
            images_name.append(i_name)
            hub_x = int(row[1])
            hub_y = int(row[2])
            tip_x = int(row[3])
            tip_y = int(row[4])
            hx_crop_loc.append(-1)
            hy_crop_loc.append(-1)
            hubs.append(0)

            img_path = parent_dir + "images/" + i_name 
            img = read_image(img_path)
            img = np.transpose(img.numpy(), (1,2,0))   

            msk_path = parent_dir + "masks/" + i_name 
            msk = read_image(msk_path)
            msk = np.transpose(msk.numpy(), (1,2,0))

            nhub_x = hub_x / width
            nhub_y = hub_y / height  

            if nhub_x < 0.5: 
                x_min = hub_x + 5
                x_max = x_min + (x_crop_size)
            else: 
                x_max = hub_x - 5
                x_min = x_max - (x_crop_size)

            if nhub_y < 0.5: 
                y_min = hub_y + 5
                y_max = y_min + (y_crop_size)
            else: 
                y_max = hub_y - 5
                y_min = y_max - (y_crop_size) 

            if y_min < 0: 
                y_min = 0 
                y_max = y_crop_size

            elif y_max > (height):
                y_max = height 
                y_min = y_max - y_crop_size

            if x_min < 0: 
                x_min = 0 
                x_max = x_crop_size

            elif x_max > (width):
                x_max = width 
                x_min = x_max - x_crop_size

            x2 = tip_x - x_min 
            y2 = tip_y - y_min 
            x1 = hub_x - x_min
            y1 = hub_y - y_min 
            print(i_name)
            print(x2)
            print(x1)
            
            cropped_lined_mask =  np.zeros([y_crop_size,x_crop_size], dtype =np.float32)
            
            for x_temp in range(x_crop_size):
                y_temp = y1 + round((y2 - y1) * (x_temp - x1) / (x2 - x1 + 1e-10))

                if y_temp < y_crop_size and y_temp >= 0: 
                    cropped_lined_mask[y_temp][x_temp] = 1.0

            for y_temp in range(y_crop_size): 
                x_temp = x1 + round((x2 - x1) * (y_temp - y1) / (y2 - y1 + 1e-10))

                if x_temp < x_crop_size and x_temp >= 0: 
                    cropped_lined_mask[y_temp][x_temp] = 1.0
                

            

            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

            save_path2 = parent_dir + cropped_images_dir + "/" + i_name
            cv2.imwrite(save_path2, cropped_img)


            cropped_msk = msk[y_min:y_max, x_min:x_max]

            save_path1 = parent_dir + cropped_masks_dir + "/" + i_name
            cv2.imwrite(save_path1, cropped_msk)

            save_path3 = parent_dir + cropped_masksLined_dir + "/" + i_name
            cv2.imwrite(save_path3, cropped_lined_mask)
        
    df_images_name = pd.DataFrame(images_name, columns=['Images'])
    df_hubs = pd.DataFrame(hubs, columns=['Hub Present'])
    df_hx_crop_loc = pd.DataFrame(hx_crop_loc, columns=['Hub X'])
    df_hy_crop_loc = pd.DataFrame(hy_crop_loc, columns=['Hub Y'])
    df_tx_crop_loc = pd.DataFrame(hx_crop_loc, columns=['Tip X'])
    df_ty_crop_loc = pd.DataFrame(hy_crop_loc, columns=['Tip Y'])
    
    df2 = pd.concat([df_images_name, df_hx_crop_loc, df_hy_crop_loc, df_tx_crop_loc, df_ty_crop_loc, df_hubs], axis = 1, sort = False)
    df2.to_csv(csv_file_name, index = True)



def DataModificationNoHub_WithoutHubData(parent_dir = "",
                              img_dir = "",
                              msk_dir = "",
                              y_crop_size = 256,
                              x_crop_size = 256,
                              height = 1280,
                              width = 720):



    images_path = glob.glob(parent_dir + "/" + img_dir + "/*.png")
    masks_path = glob.glob(parent_dir + "/" + msk_dir + "/*.png")

    cropped_images_dir = "images_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masks_dir = "masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masksLined_dir = "masksLined_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)

    createFolder(parent_dir + cropped_images_dir)
    createFolder(parent_dir + cropped_masks_dir)
    createFolder(parent_dir + cropped_masksLined_dir)
    
    for i in range(len(images_path)):
        
        image_path = images_path[i]
        msk_path = masks_path[i]
        
        print(image_path)
        print(msk_path)
        
        img = np.transpose(read_image(image_path).type(torch.FloatTensor).numpy(),(1,2,0))
        msk = np.transpose(read_image(masks_path[i]).type(torch.FloatTensor).numpy(),(1,2,0))[:,:,0]
        
        i_name = os.path.basename(images_path[i])
        print(i_name)
        
        x = []
        y = []
        
        for j in range(len(msk)): 
            for k in range(len(msk[0])):
                if msk[j][k] > 0:
                    y.append(j)
                    x.append(k)
                    
        if len(y) != 0: 
            
            y_mean = random.randint(int(sum(y)/len(y)) - 100, int(sum(y)/len(y)) - 50)
            x_mean = random.randint(int(sum(x)/len(x)) - 50, int(sum(x)/len(x)) + 50) 
    
            y_min = y_mean - int(y_crop_size/2)
            x_min = x_mean - int(x_crop_size/2)
            y_max = y_mean + int(y_crop_size/2)
            x_max = x_mean + int(x_crop_size/2)
    
            if y_min < 0: 
                y_min = 0 
                y_max = y_crop_size
    
            elif y_max > (height):
                y_max = height 
                y_min = y_max - y_crop_size
    
            if x_min < 0: 
                x_min = 0 
                x_max = x_crop_size
    
            elif x_max > (width):
                x_max = width 
                x_min = x_max - x_crop_size 
    
            
            cropped_msk = msk[y_min:y_max, x_min:x_max]
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
            
            reg = LineFitOnMask(cropped_msk, 0.1)
    
            cropped_lined_msk = LineMaskfloat32(reg, height = y_crop_size, width = x_crop_size)
    
            save_path2 = parent_dir + cropped_images_dir + "/" + i_name
            cv2.imwrite(save_path2, cropped_img)
            
            save_path1 = parent_dir + cropped_masks_dir + "/" + i_name
            cv2.imwrite(save_path1, cropped_msk)
    
            save_path3 = parent_dir + cropped_masksLined_dir + "/" + i_name
            cv2.imwrite(save_path3, cropped_lined_msk)

            
            
def DataModification_HubMask(parent_dir = "",
                          img_dir = "",
                          msk_dir = "",
                          hub_msk_dir = "", 
                          y_crop_size = 256,
                          x_crop_size = 256,
                          height = 1280,
                          width = 720):



    images_path = glob.glob(parent_dir + "/" + img_dir + "/*.png")
    masks_path = glob.glob(parent_dir + "/" + msk_dir + "/*.png")
    hubs_mask_path = glob.glob(parent_dir + "/" + hub_msk_dir + "/*.png")
    
    cropped_images_dir = "images_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_masks_dir = "masks_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)
    cropped_hubs_dir = "masks_hub_cropped_" + str(y_crop_size) + "x" + str(x_crop_size)

    createFolder(parent_dir + "/temp/" + cropped_images_dir)
    createFolder(parent_dir + "/temp/" + cropped_masks_dir)
    createFolder(parent_dir + "/temp/" + cropped_hubs_dir)
    
    for i in range(len(hubs_mask_path)):
        
        image_path = images_path[i]
        msk_path = masks_path[i]
        hub_msk_path = hubs_mask_path[i]
        
        i_name = os.path.basename(hub_msk_path)
        
        print(image_path)
        print(msk_path)
        print(hub_msk_path)
        print(i_name)
        
        img = np.transpose(read_image(image_path).type(torch.FloatTensor).numpy(),(1,2,0))
        msk = np.transpose(read_image(msk_path).type(torch.FloatTensor).numpy(),(1,2,0))[:,:,0]
        hub_msk = np.transpose(read_image(hub_msk_path).type(torch.FloatTensor).numpy(),(1,2,0))[:,:,0]
        
        
        x = []
        y = []
        
        for j in range(len(hub_msk)): 
            for k in range(len(hub_msk[0])):
                if msk[j][k] > 0:
                    y.append(j)
                    x.append(k)
                    
        if len(y) != 0: 
            
            y_mean = random.randint(int(sum(y)/len(y)) - 100, int(sum(y)/len(y)) - 50)
            x_mean = random.randint(int(sum(x)/len(x)) - 50, int(sum(x)/len(x)) + 50) 
    
            y_min = y_mean - int(y_crop_size/2)
            x_min = x_mean - int(x_crop_size/2)
            y_max = y_mean + int(y_crop_size/2)
            x_max = x_mean + int(x_crop_size/2)
    
            if y_min < 0: 
                y_min = 0 
                y_max = y_crop_size
    
            elif y_max > (height):
                y_max = height 
                y_min = y_max - y_crop_size
    
            if x_min < 0: 
                x_min = 0 
                x_max = x_crop_size
    
            elif x_max > (width):
                x_max = width 
                x_min = x_max - x_crop_size 
    
            
            cropped_msk = msk[y_min:y_max, x_min:x_max]
            cropped_hub_msk = hub_msk[y_min:y_max, x_min:x_max]
            cropped_img = img[y_min:y_max, x_min:x_max]
            cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
            
            save_path2 = parent_dir + "/temp/" + cropped_images_dir + "/" + i_name
            cv2.imwrite(save_path2, cropped_img)
            
            save_path1 = parent_dir + "/temp/" + cropped_masks_dir + "/" + i_name
            cv2.imwrite(save_path1, cropped_msk)
    
            save_path3 = parent_dir + "/temp/" + cropped_hubs_dir + "/" + i_name
            cv2.imwrite(save_path3, cropped_hub_msk)

        

    
    
def DataModificationPredMask(model_name,
    parent_dir,
    data_folder,
    sub_folder,
    img_dir,
    msk_dir, 
    csv_filename):

    model = torch.jit.load(parent_dir + "/models/shaft_segmentation_models/" + model_name)
    df = pd.read_csv(parent_dir + "/" + data_folder + "/" + sub_folder +"/" + csv_filename)
    
    images_path = []
    imgMask_names = [] 

    for name in df["Images"]: 
        images_path.append(parent_dir + "/" + data_folder + "/" + sub_folder + "/" + img_dir + "/" + name)
        imgMask_names.append(name)

    new_mask_folder = parent_dir + "/" + data_folder + "/" + sub_folder + "/" + msk_dir + "_pred"  
    createFolder(new_mask_folder)

    for n in range(len(imgMask_names)):
            
        rgb_input_tensor = read_image(images_path[n], mode=ImageReadMode.RGB)
        _, height, width = rgb_input_tensor.size() 
        input_tensor = rgb_input_tensor.type(torch.FloatTensor)
        input_tensor = input_tensor / 255 
        input_tensor = input_tensor.to(device)
        input_tensor = torch.reshape(input_tensor,(1,3,height,width))

        with torch.no_grad(): 
            pred_mask = model.forward(input_tensor)
                
        pred_label = np.transpose(pred_mask[0].cpu(),(1,2,0)).numpy()
        #Y, X = np.where(pred_label[:,:,0] > 0.3)

        #for i in range(len(Y)):
        #    r_float = random.random() 
        #    if (r_float > 0.2):
        #        pred_label[Y[i],X[i],0] = 0.0

        
        cv2.imwrite(new_mask_folder + "/" + imgMask_names[n], pred_label)   


def DataModificationPredMaskWithNoise(model_name,
    parent_dir,
    data_folder,
    sub_folder,
    img_dir,
    msk_dir, 
    csv_filename):

    model = torch.jit.load(parent_dir + "/models/shaft_segmentation_models/" + model_name)
    df = pd.read_csv(parent_dir + "/" + data_folder + "/" + sub_folder +"/" + csv_filename)
    
    images_path = []
    imgMask_names = [] 
    hub_y = []
    hub_x = []


    for name, hubY, hubX in zip(df["Images"],  df["Hub Y"], df["Hub X"]): 
        images_path.append(parent_dir + "/" + data_folder + "/" + sub_folder + "/" + img_dir + "/" + name)
        imgMask_names.append(name)
        hub_y.append(hubY)
        hub_x.append(hubX)

    new_mask_folder = parent_dir + "/" + data_folder + "/" + sub_folder + "/" + msk_dir + "_predNoise"  
    createFolder(new_mask_folder)

    for n in range(len(imgMask_names)):
            
        rgb_input_tensor = read_image(images_path[n], mode=ImageReadMode.RGB)
        _, height, width = rgb_input_tensor.size() 
        input_tensor = rgb_input_tensor.type(torch.FloatTensor)
        input_tensor = input_tensor / 255 
        input_tensor = input_tensor.to(device)
        input_tensor = torch.reshape(input_tensor,(1,3,height,width))

        with torch.no_grad(): 
            pred_mask = model.forward(input_tensor)
                
        pred_label = np.transpose(pred_mask[0].cpu(),(1,2,0)).numpy()
        Y, X = np.where(pred_label[:,:,0] > 0.3)

        h_y = hub_y[n]
        h_x = hub_x[n]

        a1 = random.randint(20,50)
        a2 = random.randint(50,100)
 


        for i in range(len(Y)):
            
            dis = np.sqrt(pow(Y[i]-h_y,2) + pow(X[i]-h_x,2))
            r_float = random.random() 

            if dis > a1 and dis < a2: 
                pred_label[Y[i],X[i],0] = 0.0
            elif (r_float > 0.75):
                pred_label[Y[i],X[i],0] = 0.0

        
        cv2.imwrite(new_mask_folder + "/" + imgMask_names[n], pred_label) 
                         
            
if __name__ == "__main__": 
    

    folders = [ "PB","SS"]
    
    txt_filename = "groundtruth_2point.txt"
                             

    y_crop_size = 256
    x_crop_size = 256
    height = 1280
    width = 720
    
    ifSaveImages = True
    ifSaveCSV = True


   
    for sub_folder in folders: 
        
        parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena_BD_25G/" + sub_folder + "/"
                             
#         DataModification_HubMask(parent_dir,
#                                 img_dir = "images",
#                                 msk_dir = "masks",
#                                 hub_msk_dir = "masks_hub",
#                                 y_crop_size = 256,
#                                 x_crop_size = 256,
#                                 height = 1280,
#                                 width = 720)
        
#         dataModificationOriginal(parent_dir = parent_dir, 
#                                 txt_filename = txt_filename,
#                                 y_crop_size = y_crop_size,
#                                 x_crop_size = x_crop_size, 
#                                 height = height, 
#                                 width = width, 
#                                 ifSaveImages = ifSaveImages, 
#                                 ifSaveCSV = ifSaveCSV)
#         dataModification(parent_dir = parent_dir, 
#                                 txt_filename = txt_filename,
#                                 y_crop_size = y_crop_size,
#                                 x_crop_size = x_crop_size, 
#                                 height = height, 
#                                 width = width, 
#                                 ifSaveImages = ifSaveImages, 
#                                 ifSaveCSV = ifSaveCSV)
        
#         dataModificationNoHub(parent_dir = parent_dir,
#                              txt_filename = txt_filename,
#                              y_crop_size = y_crop_size,
#                              x_crop_size = x_crop_size,
#                              height = height, 
#                              width = width)

#         DataModificationNoHub_WithoutHubData(parent_dir = parent_dir,
#                              img_dir = "images",
#                              msk_dir = "masks",                
#                              y_crop_size = y_crop_size,
#                              x_crop_size = x_crop_size,
#                              height = height, 
#                              width = width)
        
        
        dataModificationWithoutSegmentation(parent_dir = parent_dir,
                    txt_filename = txt_filename,
                    y_crop_size = y_crop_size,
                    x_crop_size = x_crop_size,
                    height = height,
                    width = width,
                    ifSaveCSV = ifSaveCSV,
                    ifSaveImages = ifSaveImages)


#         dataModification(parent_dir = parent_dir,
#                     txt_filename = txt_filename,
#                     y_crop_size = y_crop_size,
#                     x_crop_size = x_crop_size,
#                     height = height,
#                     width = width,
#                     ifSaveCSV = ifSaveCSV,
#                     ifSaveImages = ifSaveImages)

        
            