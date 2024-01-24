"""
Training for hub regression

__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""
import torch
import torch.nn as nn
import time 
from time import sleep
import datetime
import os 
import cv2
import numpy as np 
import torch.optim as optim
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
import pandas as pd
import TrainHelper as th
import CustomModels as cm
import CustomHubModels as cmhub
import CustomImageDataset as cid
import torch.nn.functional as F
import LossFunction as lf 
import HubLossFunctions as hlf 
from torchvision.io import read_image, ImageReadMode
import random
device = ("cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append(os.path.join('..','segment-utils'))
import VideoAnnotationHelper as vah
# import visualize_model as visualize

#!pip install torchinfo
from torchinfo import summary
import torchinfo

random.seed(101)
def run_training():
    parent_dir = "C:/Lumena"
    train_data_folder = "train_data"
    sub_folders = ["EL"]#, "PC2", "PR", "EL_2", "CNMC", "CNMC.2023.11.07", "SS"]
    csv_filename = ["cropped_256x256.csv","cropped_256x256_original.csv"]
    img_dir = ["images_cropped_256x256", "images_cropped_256x256_original"]
    mask_dir = ["masks_cropped_256x256","masks_cropped_256x256_original"]

    batch_size = 64
    epochs = 1000
    cycles = 1 

    encoder_lr = 1e-4
    regA_lr = 1e-3
    regB_lr = 1e-4
    
    stop_encoder_requires_grad = False  
    stop_regA_requires_grad = False 
    stop_regB_requires_grad = True 

    FullRange = False 
    Gray = False
    ifTrain = True 
    ifMSKCha = False 

    observe_path="D:/test_dir"

    old_model_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/shaft_segmentation/model_cyc_1_ep_200_bs_16_lr_0.0001_20231116_T160049.pt"
    org_r18model_path = "Resnet18_Imagenet_Weights.pt"


    images_path = []
    labels = []

    for k in range(len(csv_filename)):
        dfs = []
        for ele in sub_folders:
            df = pd.read_csv(parent_dir + "/" + train_data_folder + "/" + ele +"/" + csv_filename[k])
            dfs.append(df)

        for i in range(len(dfs)): 
            # for name in zip(dfs[i]["Images"], dfs[i]["Hub Y"], dfs[i]["Hub X"], dfs[i]["Hub Present"]): 
            for name in zip(dfs[i]["Images"], dfs[i]["Hub Y"], dfs[i]["Hub X"], dfs[i]["Tip Y"], dfs[i]["Tip X"], dfs[i]["Hub Present"]): 
                if name[3]:
                    label_ele = [] 
                    images_path.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + img_dir[k] + "/" + name[0])
                    label_ele.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + mask_dir[k] + "/" + name[0])
                    label_ele.append(name[1:])
                    labels.append(label_ele)
        
        
    X_train, X_val, y_train, y_val = train_test_split(images_path, labels, test_size = 0.2, random_state = 24)

    # train_dataset = cid.Custom_ImageRegressor_Dataset(X_train, y_train, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange, ifAddMskCha = ifMSKCha)
    # val_dataset = cid.Custom_ImageRegressor_Dataset(X_val, y_val, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange, ifAddMskCha = ifMSKCha)
    train_dataset = cid.Custom_ImageRegressor_Dataset2(X_train, y_train, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange, ifAddMskCha = ifMSKCha)
    val_dataset = cid.Custom_ImageRegressor_Dataset2(X_val, y_val, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange, ifAddMskCha = ifMSKCha)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,num_workers=1,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False,num_workers=1,pin_memory = True)

    print("A")

    model_name = "D7_res18_lean_hubtiplineloss" 
    model_name += "_cyc_" + str(cycles) + "_ep_" + str(epochs) + "_bs_" + str(batch_size)
    model_name += "_encoder_lr_" + str(encoder_lr)
    model_name += "_lr_" + str(regA_lr)
    model_folder = model_name +  "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now())
    model_name_dt = model_folder
    
    vah.createFolder(observe_path + "/"  + model_folder)

    old_model = torch.jit.load(old_model_path)
    summary(old_model)

    # model = cm.ConvEncoderRegressor(ifFullRange = FullRange).to(device)
    # model = cmhub.ResnetEncoderRegressor_Lean(ifFullRange = FullRange).to(device)
    model = cmhub.ResnetEncoderRegressorHubTIp(ifFullRange = FullRange).to(device)
    
    # summary(model)
    torchinfo.summary(model, [(1, 3, 256, 256)] 
                            ,col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable")
                            # ,verbose = 1
                            )
    # for param in model.parameters():
    #     print(param.data)
    #     print("------------------------")
        

    # # model.load_state_dict(old_model.state_dict(), strict=False)
    # org_r18model = torch.jit.load(org_r18model_path)
    # model.load_state_dict(org_r18model.state_dict(), strict=False)
    
    # for param in model.parameters():
    #     print(param.data)
    #     print("------------------------")

    # model_script = torch.jit.script(model)
    # model_script.save("Resnet18_Random_Weights.pt") # Resnet18_Random_Weights.pt Resnet18_Zero_Weights.pt Resnet18_Imagenet_Weights
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("NO.", i, name)
            i += 1

    lossFun1 = nn.MSELoss()
    lossFun2 = hlf.LineDistanceLoss()

    i = 0 
    encoder_list = []
    regressorA_list = []
    regressorB_list = [] 

    for param in model.parameters(): 
        if i < 60:# 8 
            if stop_encoder_requires_grad:
                param.requires_grad = False 
            encoder_list.append(param)
        elif i < 70:# 16 
            if stop_regA_requires_grad: 
                param.requires_grad = False 
            regressorA_list.append(param)
        else: 
            if stop_regB_requires_grad: 
                param.requires_grad = False 
            regressorB_list.append(param)
            
        i += 1


    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)


    
    print("B")

    for j in range(cycles):

        print("Training Cycle: ", j+1)

        optimizer = torch.optim.Adam([{"params":encoder_list, "lr":encoder_lr},
                                    {"params":regressorA_list, "lr":regA_lr},
                                    #{"params":regressorB_list, "lr":regB_lr}
                                    ], weight_decay = 1e-5)
        
        
    #     early_stopping = th.EarlyStopping(patience = 10, min_delta = 0.001)
        
        min_val_loss = []
        save_model = True
        for epoch in range(epochs): 
            
            train_loss = 0.0
            val_loss = 0.0        
            start_time = time.time() 
            
            model.train()
            for data in train_loader:
                images, true_yx, true_p = data 
                timex = time.process_time()
                images = images.to(device)
                true_yx = true_yx.to(device)

                optimizer.zero_grad()
                
                pred_yx, pred_p = model(images)

                hubloss = lossFun1.forward(pred_yx[:,0:2].float(),true_yx[:,0:2].float()) # MSE only for the hub for now. TODO: Add a new custom MSE with highter weight for the hub
                tiploss = lossFun1.forward(pred_yx[:,2:].float(),true_yx[:,2:].float())
                lineloss = lossFun2.forward(pred_yx.float(),true_yx.float())
                loss = hubloss + 0.5*tiploss + lineloss
                loss.backward()
                optimizer.step()
                
                train_loss += loss

                
            model.eval()    
            for data in val_loader: 
                
                images, true_yx, true_p= data
                images = images.to(device)
                true_yx = true_yx.to(device)

                with torch.no_grad(): 
                    pred_yx, pred_p = model(images)
                    hubloss = lossFun1.forward(pred_yx[:,0:2].float(),true_yx[:,0:2].float()) 
                    tiploss = lossFun1.forward(pred_yx[:,2:].float(),true_yx[:,2:].float())
                    lineloss = lossFun2.forward(pred_yx.float(),true_yx.float())
                    loss = hubloss + 0.5*tiploss + lineloss
                    # loss1 = lossFun1.forward(pred_yx,true_yx)
        
                val_loss += loss

            
            train_loss = train_loss/len(train_loader)    
            val_loss = val_loss/len(val_loader)
            
            
            print('Epoch: {}  \tTraining Loss: {:.6f} \t Vali Loss: {:.6f} \t Exe Time: {:.6f}sec'.format(
            epoch, 
            train_loss, 
            val_loss,
            time.time() - start_time))

    #         stop = early_stopping.early_stop(val_loss, model)
    #         if stop: 
    #             model = early_stopping.get_model() 
    #             print("Early Stopping because of regressor val loss")
    #             break 
            if (epoch == 0):
                min_val_loss = val_loss
            else:
                if val_loss < min_val_loss:
                    save_model = True
                    min_val_loss = val_loss
                else: 
                    save_model = False

            if save_model == True:#epoch % 10 == 0: 
                model_script = torch.jit.script(model)
                # temp_name = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
                temp_name = "chkpt_ep{}_".format(epoch) + model_name_dt + ".pt"
                model_script.save(observe_path + "/"  + model_folder + "/" + temp_name)
                print("Temporary model saved: ", temp_name)     
            if epoch % 10 == 0: 
                model_script = torch.jit.script(model)
                temp_name = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
                model_script.save(observe_path + "/"  + model_folder + "/" + temp_name)
                print("Temporary model saved: ", temp_name)           

    model_script = torch.jit.script(model)
    # temp_name = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
    temp_name = model_name_dt + "_final" + ".pt"
    model_script.save(observe_path + "/"  + model_folder + "/" + temp_name)
    print("Temprary model saved: ", temp_name)

if __name__ == '__main__':
    from torchvision import datasets, transforms, models
    from collections import OrderedDict

    # print("---------------------resnet18-------------------------------")
    # model = models.resnet18(pretrained=True)
    # #print(model)
    # torchinfo.summary(model, [(1, 3, 256, 256)] 
    #                         ,col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable")
    #                         # ,verbose = 1
    #                         ) 
    # summary(model)
    # print("----------------------------------------------------")

    # print("---------------------resnet18 fc removed-------------------------------")
    # newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    # #print(newmodel)
    # torchinfo.summary(newmodel, [(1, 3, 256, 256)] 
    #                         ,col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable")
    #                         # ,verbose = 1
    #                         ) 
    # print("----------------------------------------------------")

    # print("---------------------resnet18 fc removed ordered dict-------------------------------")
    # newmodel_samenames = torch.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
    # # print(newmodel_samenames)
    # torchinfo.summary(newmodel_samenames, [(1, 3, 256, 256)] 
    #                         ,col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable")
    #                         # ,verbose = 1
    #                         )
    # print("----------------------------------------------------")

    run_training()
    print("training done")