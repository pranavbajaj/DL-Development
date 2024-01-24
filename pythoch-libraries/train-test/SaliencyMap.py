import torch
import torch.nn as nn
import os 
import glob 
from torch.utils.data import DataLoader 
import pandas as pd
import sys 
sys.path.append(os.path.join('..','segment-utils'))
from VideoAnnotationHelper import createFolder

device = ("cuda" if torch.cuda.is_available() else "cpu")





def segmentation_saliency_map(model_path = "",
                              folder_path = "",
                              input_type = ".png",
                              output_folder_path = ""):
    
    
    model = torch.jit.load(model_path)
    
    createFolder(output_folder_path)
    
    image_paths =  glob.glob(input_path + "/*" + input_type)
  
    
    for img_path in image_paths: 

        img_name = os.path.basename(img_path)
        
        
        rgb_input_tensor = read_image(img_path,mode=ImageReadMode.RGB)
        img_numpy = np.transpose(rgb_input_tensor.numpy(),(1,2,0))
        img = rgb_input_tensor.type(torch.FloatTensor)
        img = img / 255
        img = img.to(device) 
        
        img.requires_grad = True 
    
        _, height_, width_ = input_tensor.size() 
        
        pred = model.forward(torch.reshape(img,(1,3,height_,width_)))
        scores = torch.sum(pred, dim=(2,3))
        scores.backward()

        saliency, _ = torch.max(imgs.grad.data.abs(),dim=1)
        saliency_numpy = saliency.detach().cpu().numpy()
        
        
        
        
def saliency_map(): 
    
        
        