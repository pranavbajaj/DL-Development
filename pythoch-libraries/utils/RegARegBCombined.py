"""

Custom Autoencoders 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""


import sys
import os
sys.path.append(os.path.join('..','segment-utils'))
sys.path.append(os.path.join('..','segment-pytorch'))
import torch
import torch.nn as nn 
import time
import datetime
device = ("cuda" if torch.cuda.is_available() else "cpu")



class RegARegBCombinedModel(nn.Module):
    
    def __init__(self, original_model): 
        super(RegARegBCombinedModel, self).__init__()
        
        self.layer = original_model
        
    def forward(self,x):
        return torch.cat(self.layer(x), -1)

    
    
def ModelConvert(original_model_path = "",
                 new_model_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/hub_regression/"):
    
    # Loading the original model
    original_model =  torch.jit.load(original_model_path)
    
    # Initializing the new model
    new_model = RegARegBCombinedModel(original_model).to(device)
    
    # Saving the new model
    model_script = torch.jit.script(new_model)
    model_name  = "CombinedRegARegB_{:%Y%m%d_T%H%M%S}".format(datetime.datetime.now()) + ".pt"
    print(model_name)
    # Saving The new model 
    model_script.save(new_model_path + "/" + model_name)
    
    
    
    
    
    
if __name__ == "__main__": 
    
    print("Transfering the model...")
    
    original_model_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/hub_regression/model_cyc_1_ep_400_bs_128_lr_1e-05_20240118_T115016.pt"
    new_model_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/models/hub_regression/"

    ModelConvert(original_model_path, new_model_path)
    
    
    
    
    