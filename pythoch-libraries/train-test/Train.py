import torch 
import torch.nn as nn
import time 
from time import sleep
import datetime
import os 
import numpy as np 
import torch.optim as optim
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
import pandas as pd
import TrainHelper as th
import CustomModels as cm
import CustomImageDataset as cid
import LossFunction as lf 

sys.path.append(os.path.join('..','segment-utils'))
import VideoAnnotationHelper as vah

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def createFolder(path="D:/lol"):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder '{path}'.")
    else:
        print(f"Folder '{path}' already exists.")

        
def TainShaftSegmentation(fineTune = False, model_path = "", batch_size = 32, learning_rate = 1e-2, epochs = 1000, cycles = 3, parent_dir="", train_data_folder = "train_data", sub_folders=["EL"], 
img_dir = "images_cropped", mask_dir = "masks_cropped", csv_filename = "cropped.csv", observe_path="D:/test_dir"):
      
    print(learning_rate)
    init_lr = learning_rate
    dfs = []
    for ele in sub_folders:
        df = pd.read_csv(parent_dir + "/" + train_data_folder + "/" + ele +"/" + csv_filename)
        dfs.append(df)
        
    images_path = []
    masks_path = []
    
    for i in range(len(dfs)): 
        for name in dfs[i]["Images"]: 
            images_path.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + img_dir + "/" + name)
            masks_path.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + mask_dir + "/" + name)
            
    print("Number of Images: ", len(images_path))
    print("Number of masks: ", len(masks_path))
            
    X_train, X_val, y_train, y_val = train_test_split(images_path, masks_path, test_size = 0.2,random_state = 42)
    
    train_dataset = cid.CustomImageGrayMaskDataset_dataAugmentation(X_train, y_train, ifgray = False)
    val_dataset = cid.CustomImageGrayMaskDataset_dataAugmentation(X_val, y_val, ifgray = False)
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    if fineTune: 
        model = torch.jit.load(model_path)
    else: 
        model = cm.ConvAutoEncoder_WithSkipConnections3().to(device)
        
    
    lossFun = lf.IoULoss() 
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_name = "model" 
    model_name += "_cyc_" + str(cycles) + "_ep_" + str(epochs) + "_bs_" + str(batch_size)
    model_name += "_lr_" + str(init_lr)
    model_folder = model_name +  "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now())

    createFolder(observe_path + "/"  + model_folder)
    
    for j in range(cycles):
        
        print("Training Cycle: ", j+1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        learning_rate /= 10 
        early_stopping = th.EarlyStopping(patience = 50, min_delta = 0.01)

        for epoch in range(epochs): 
            train_loss = 0.0
            val_loss = 0.0
            
            start_time = time.time() 
           
            for data in train_loader: 
                
                #model.train() 
                images, labels = data 
                images = images.to(device)
                labels= labels.to(device)
                optimizer.zero_grad()
                output = model(images)

                loss = lossFun.forward(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            
            for data in val_loader: 
                
                #model.eval() 
                images, labels = data
                images = images.to(device)
                labels= labels.to(device)
                with torch.no_grad(): 
                    outputs = model(images)
                    loss = lossFun.forward(outputs,labels)
                
                val_loss += loss.item()
                  
            train_loss = train_loss/len(train_loader)    
            val_loss = val_loss/len(val_loader)

            print('Epoch: {}  \tTraining Loss: {:.6f} \t Vali Loss: {:.6f} \t Exe Time: {:.6f}sec'.format(
            epoch, 
            train_loss, 
            val_loss,
            time.time() - start_time))
            
            #stop = early_stopping.early_stop(val_loss, model)
            #if stop: 
            #    model = early_stopping.get_model() 
            #    print("Early Stopping")
            #    break 
                       
            #if (val_loss - train_loss) > 0.05: 
            #    print("Stopping Training to avoid Overfitting")
            #    break 

            if epoch % 50 == 0: 

                model_script = torch.jit.script(model)
                temp_name = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
                model_script.save(observe_path + "/"  + model_folder + "/" + temp_name)
                print("Temprary model saved: ", temp_name)

            
           
                
    
    model_script = torch.jit.script(model)
    
      
    final_model = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now())
    
    if fineTune: 
        final_model += "finetuned.pt"
    else: 
        final_model += ".pt"
        
    model_script.save("//cgmqnap.clearguide.local/data/Needles/Lumena" + "/models/shaft_segmentation_models/" + final_model)
    print("Saved trained model: ", final_model)
    
    
    
def TrainShaftHub(): 
    
    # Dir of Train data 
    parent_dir = "C:/Lumena"
    train_data_folder = "train_data"
    sub_folders = ["EL", "PC2", "PR", "EL_2"]
    csv_filename = "original.csv"
    img_dir = "images"
    mask_dir = "masks"

    # Hyperparameters 
    batch_size = 6 
    epochs = 200
    cycles = 1 
    encoder_lr = 1e-5 
    decoder_lr = 1e-5 
    regA_lr = 1e-4
    regB_lr = 1e-4
    testTrainSplit = 0.2 

    # Freezing the layers 
    stop_encoder_requires_grad = False  
    stop_decoder_requires_grad = True 
    stop_regA_requires_grad = False 
    stop_regB_requires_grad = True 

    # Type of data 
    FullRange = False 
    Gray = False
    ifTrain = True 

    # Path to store intermedial model. 
    observe_path="D:/test_dir"

    

    # Data loading 
    dfs = []
    for ele in sub_folders:
        df = pd.read_csv(parent_dir + "/" + train_data_folder + "/" + ele +"/" + csv_filename)
        dfs.append(df)
    images_path = []
    labels = []
    for i in range(len(dfs)): 
        for name in zip(dfs[i]["Images"], dfs[i]["Hub Y"], dfs[i]["Hub X"], dfs[i]["Hub Present"]): 
            if name[3]:
                label_ele = [] 
                images_path.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + img_dir + "/" + name[0])
                label_ele.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + mask_dir + "/" + name[0])
                label_ele.append(name[1:])
                labels.append(label_ele)
            
    X_train, X_val, y_train, y_val = train_test_split(images_path, labels, test_size = testTrainSplit)
    train_dataset = cid.Custom_ImageGrayMaskRegressor_Dataset(X_train, y_train, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange)
    val_dataset = cid.Custom_ImageGrayMaskRegressor_Dataset(X_val, y_val, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,num_workers=1,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False,num_workers=1,pin_memory = True)

    # Create a folder inside the observe_path to save temp model. 
    model_name = "model" 
    model_name += "_cyc_" + str(cycles) + "_ep_" + str(epochs) + "_bs_" + str(batch_size)
    model_name += "_lr_" + str(regA_lr)
    model_folder = model_name +  "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now())
    createFolder(observe_path + "/"  + model_folder)

    # Path to shaft segmentation model whose trained weights are used to train the segmentation-regression model.  
    old_model_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/shaft_segmentation_models/model_cyc_3_ep_1000_bs_16_lr_1e-05_20230929_T060507.pt"
    # Loading the segmentation model 
    old_model = torch.jit.load(old_model_path)

    # Loading the segmentation-regression model 
    model = cm.ConvAutoEncoder_WithRegressor(ifFullRange = FullRange).to(device)
    # Transfering weights from seg model to seg-reg model 
    model.load_state_dict(old_model.state_dict(), strict=False)

    i = 0 
    encoder_list = []
    decoder_list = []
    regressorA_list = []
    regressorB_list = [] 

    for param in model.parameters(): 
        if i < 8: 
            if stop_encoder_requires_grad:
                param.requires_grad = False 
            encoder_list.append(param)
        elif i < 16: 
            if stop_decoder_requires_grad: 
                param.requires_grad = False 
            decoder_list.append(param)
        elif i < 24: 
            if stop_regA_requires_grad: 
                param.requires_grad = False 
            regressorA_list.append(param)
        else: 
            if stop_regB_requires_grad: 
                param.requires_grad = False 
            regressorB_list.append(param)
        
        i += 1

    
    # Loss 
    lossFun1 = nn.MSELoss()

    # Training 
    for j in range(cycles):

        print("Training Cycle: ", j+1)

        optimizer = torch.optim.Adam([{"params":encoder_list, "lr":encoder_lr},
                                      #{"params":decoder_list, "lr":decoder_lr},
                                      {"params":regressorA_list, "lr":regA_lr},
                                      #{"params":regressorB_list, "lr":regB_lr}
                                     ])
    
        early_stopping = th.EarlyStopping(patience = 10, min_delta = 0.001)
    
        for epoch in range(epochs): 
        
            train_loss = 0.0
            val_loss = 0.0        
            start_time = time.time() 
        
            model.train()
            for data in train_loader:
                images, labels, true_yx, true_p= data 
                timex = time.process_time()
                images = images.to(device)
                true_yx = true_yx.to(device)

                optimizer.zero_grad()
            
                outputs, pred_yx, pred_p = model(images)

                loss1 = lossFun1.forward(pred_yx.float(),true_yx.float())
                loss1.backward()
                optimizer.step()
            
                train_loss += loss1

            
            model.eval()    
            for data in val_loader: 
            
                images, labels, true_yx, true_p= data
                images = images.to(device)
                true_yx = true_yx.to(device)

                with torch.no_grad(): 
                    outputs, pred_yx, pred_p = model(images)
                    loss1 = lossFun1.forward(pred_yx,true_yx)
     
                val_loss += loss1

        
            train_loss = train_loss/len(train_loader)    
            val_loss = val_loss/len(val_loader)
        
        
            print('Epoch: {}  \tTraining Loss: {:.6f} \t Vali Loss: {:.6f} \t Exe Time: {:.6f}sec'.format(
            epoch, 
            train_loss, 
            val_loss,
            time.time() - start_time))

            stop = early_stopping.early_stop(val_loss, model)
            if stop: 
                model = early_stopping.get_model() 
                print("Early Stopping because of regressor val loss")
                break 
        
            if epoch % 10 == 0: 
 
                model_script = torch.jit.script(model)
                temp_name = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
                model_script.save(observe_path + "/"  + model_folder + "/" + temp_name)
                print("Temprary model saved: ", temp_name)            
            
        encoder_lr /= encoder_lr
        decoder_lr /= decoder_lr
        regA_lr /= regA_lr
        regB_lr /= regB_lr

    
    # Saving the final model. 
    model_script = torch.jit.script(model)
    final_model = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now())
    final_model += "regression.pt"
        
    model_script.save("//cgmqnap.clearguide.local/data/Needles/Lumena" + "/models/shaft_segmentation_models/" + final_model)
    print("Saved trained model: ", final_model)


def TrainHubRegression(parent_dir="C:/Lumena",
    train_data_folder = "train_data",
    sub_folders = ["EL", "PC2", "PR", "EL_2"],
    csv_filename = ["cropped_456x256.csv","cropped_456x256_original_.csv"],
    img_dir = ["images_cropped_456x256", "images_cropped_456x256_original"],
    mask_dir = ["masks_cropped_456x256","masks_cropped_456x256_original"],
    batch_size = 32, 
    epochs = 200,
    cycles = 1, 
    encoder_lr = 1e-4,
    regA_lr = 1e-4, 
    regB_lr = 1e-4, 
    stop_encoder_requires_grad = False, 
    stop_regA_requires_grad = False, 
    stop_regB_requires_grad = False, 
    FullRange = False, 
    Gray = False, 
    ifTrain = True, 
    observe_path = "D:/test_dir", 
    ifFineTune = False, 
    old_model_path = ""):

    images_path = []
    labels = []

    for k in range(len(csv_filename)):
        dfs = []
        for ele in sub_folders:
            df = pd.read_csv(parent_dir + "/" + train_data_folder + "/" + ele +"/" + csv_filename[k])
            dfs.append(df)

        for i in range(len(dfs)): 
            for name in zip(dfs[i]["Images"], dfs[i]["Hub Y"], dfs[i]["Hub X"], dfs[i]["Hub Present"]): 
                if name[3]:
                    label_ele = [] 
                    images_path.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + img_dir[k] + "/" + name[0])
                    label_ele.append(parent_dir + "/" + train_data_folder + "/" + sub_folders[i] + "/" + mask_dir[k] + "/" + name[0])
                    label_ele.append(name[1:])
                    labels.append(label_ele)
            
    X_train, X_val, y_train, y_val = train_test_split(images_path, labels, test_size = 0.2)

    train_dataset = cid.Custom_ImageRegressor_Dataset(X_train, y_train, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange)
    val_dataset = cid.Custom_ImageRegressor_Dataset(X_val, y_val, ifgray = Gray, ifTrain = ifTrain, ifFullRange = FullRange)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,num_workers=1,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False,num_workers=1,pin_memory = True)


    model_name = "model" 
    model_name += "_cyc_" + str(cycles) + "_ep_" + str(epochs) + "_bs_" + str(batch_size)
    model_name += "_lr_" + str(regA_lr)
    model_folder = model_name +  "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now())
 
    createFolder(observe_path + "/"  + model_folder)

    model = cm.ConvEncoderRegressor(ifFullRange = FullRange).to(device)

    if ifFineTune: 
        old_model = torch.jit.load(old_model_path)
        model.load_state_dict(old_model.state_dict(), strict=False)

    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("NO.", i, name)
            i += 1

    lossFun1 = nn.MSELoss()

    i = 0 
    encoder_list = []
    regressorA_list = []
    regressorB_list = [] 

    for param in model.parameters(): 
        if i < 8: 
            if stop_encoder_requires_grad:
                param.requires_grad = False 
            encoder_list.append(param)
        elif i < 16: 
            if stop_regA_requires_grad: 
                param.requires_grad = False 
            regressorA_list.append(param)
        else: 
            if stop_regB_requires_grad: 
                param.requires_grad = False 
            regressorB_list.append(param)
        
        i += 1

    for j in range(cycles):

        print("Training Cycle: ", j+1)

        optimizer = torch.optim.Adam([{"params":encoder_list, "lr":encoder_lr},
                                      {"params":regressorA_list, "lr":regA_lr
                                     ])
    
        for epoch in range(epochs): 
        
            train_loss = 0.0
            val_loss = 0.0        
            start_time = time.time() 
        
            model.train()
            for data in train_loader:
                images, true_yx, true_p= data 
                timex = time.process_time()
                images = images.to(device)
                true_yx = true_yx.to(device)

                optimizer.zero_grad()
            
                pred_yx, pred_p = model(images)

                loss1 = lossFun1.forward(pred_yx.float(),true_yx.float())
                loss1.backward()
                optimizer.step()
            
                train_loss += loss1

            model.eval()    
            for data in val_loader: 
            
                images, true_yx, true_p= data
                images = images.to(device)
                true_yx = true_yx.to(device)

                with torch.no_grad(): 
                    pred_yx, pred_p = model(images)
                    loss1 = lossFun1.forward(pred_yx,true_yx)
     
                val_loss += loss1

            train_loss = train_loss/len(train_loader)    
            val_loss = val_loss/len(val_loader)

            print('Epoch: {}  \tTraining Loss: {:.6f} \t Vali Loss: {:.6f} \t Exe Time: {:.6f}sec'.format(
            epoch, 
            train_loss, 
            val_loss,
            time.time() - start_time))

            if epoch % 10 == 0: 
                model_script = torch.jit.script(model)
                temp_name = model_name + "_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
                model_script.save(observe_path + "/"  + model_folder + "/" + temp_name)
                print("Temprary model saved: ", temp_name)        

            
if __name__ == "__main__": 
    
    print("Traning")
    TrainShaftHub() 

    #fineTune = True 
    #model_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/shaft_segmentation_models/model_cyc_3_ep_1000_bs_16_lr_0.0001_20230926_T174052.pt"
    #batch_size = 16
    #learning_rate = 1e-4 
    #epochs = 500 
    #cycles = 3 
    #parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena"
    #parent_dir = "D:/Lumena"
    #train_data_folder = "train_data"
    #sub_folders = ["EL", "PC2", "PR"]
    #img_dir = "images_cropped"
    #mask_dir = "masks_cropped"
    #csv_filename = "cropped.csv"
    #observe_path="D:/test_dir" 
    
    #TainShaftSegmentation(fineTune = fineTune, 
    #    model_path = model_path, 
    #    batch_size = batch_size, 
    #    learning_rate = learning_rate, 
    #    epochs = epochs, 
    #    cycles = cycles, 
    #    parent_dir = parent_dir, 
    #    train_data_folder = train_data_folder , 
    #    sub_folders = sub_folders, 
    #    img_dir = img_dir, 
    #    mask_dir = mask_dir, 
    #    csv_filename = csv_filename, 
    #    observe_path=observe_path )                                                 
    
   
    
 