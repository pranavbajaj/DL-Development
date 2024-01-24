"""
Hub regression models

__author__ = "P Rajan"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from collections import OrderedDict
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class ResnetEncoderRegressor(nn.Module):
    
    def __init__(self, ifFullRange = False): 
        super(ResnetEncoderRegressor, self).__init__()
        
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # model = models.resnet18()
        self.resnet18 = torch.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
        
        # Regressor A  # 57344
        self.fca1 = nn.Linear(32768,512)
        self.fca2 = nn.Linear(512,256)
        self.fca3 = nn.Linear(256,32)
        self.fca4 = nn.Linear(32,2)
        self.fca5 = nn.Linear(256,32)

        # Regressor B
        self.fcb1 = nn.Linear(512,256)
        self.fcb2 = nn.Linear(256,128)
        self.fcb3_1 = nn.Linear(128,64)
        self.fcb3_2 = nn.Linear(64,32)
        self.fcb3_3 = nn.Linear(32,8)
        self.fcb4_1 = nn.Linear(8,1)
        
        # Dropout 
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.2)
        
        
        # Full range output for hubYX location
        self.ifFullRange = ifFullRange
        
        if ifFullRange: 
            print("Model with full HubYX loc")
        else: 
            print("Model with nomalized HubYX loc")

        
        
    
    def forward(self,x): 

        # Encoder 
        # x1 = F.relu(self.conv1(x)) #256x3 -> 256x16
        # x2 = self.pool(x1) # 256x16 -> 128x16
        # x3 = F.relu(self.conv2(x2)) # 128x16 -> 128x32
        # x4 = self.pool(x3) # 128x32 -> 64 x 32
        # x5 = F.relu(self.conv3(x4)) # 64x32 -> 64x64
        # x6 = self.pool(x5) # 64x64 -> 32x64 
        # x7 = F.relu(self.conv4(x6)) # 32x64 -> 32x128
        # x7 = self.pool(x7)

        x7 = self.resnet18(x)

        # Regressor A
        xa1 = torch.flatten(x7,1)
        xa1 = self.dropout(xa1)
        
        #xa2 = F.relu(self.fca1(xa1))
        #xa2 = self.dropout(xa2)
        
        xa3 = F.relu(self.fca2(xa1))
        xa3 = self.dropout(xa3)
    
        xa4 = F.relu(self.fca3(xa3))
        xa4 = self.dropout(xa4)
    
        if self.ifFullRange: 
            xa5 = self.fca4(xa4)
        else: 
            xa5 = F.sigmoid(self.fca4(xa4))

        # Regressor B
        xb2 = F.relu(self.fcb1(xa1))
        xb2 = self.dropout2(xb2)
        
        xb3 = F.relu(self.fcb2(xb2))
        xb3 = self.dropout2(xb3)
        
        xb4 = F.relu(self.fcb3_1(xb3))
        xb4 = self.dropout2(xb4)
        
        xb4 = F.relu(self.fcb3_2(xb4))
        xb4 = self.dropout2(xb4)
        
        xb4 = F.relu(self.fcb3_3(xb4))
        xb4 = self.dropout2(xb4)
        
        xb5 = F.sigmoid(self.fcb4_1(xb4))
        
        return xa5, xb5
    
class ResnetEncoderRegressor_Lean(nn.Module):
    
    def __init__(self, ifFullRange = False): 
        super(ResnetEncoderRegressor_Lean, self).__init__()
        
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # model = models.resnet18()
        self.resnet18 = torch.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
        
        # Regressor A  # 57344
        self.fca1 = nn.Linear(512,256)
        self.fca2 = nn.Linear(256,2)
        

        # Regressor B
        self.fcb1 = nn.Linear(512,256)
        self.fcb2 = nn.Linear(256,128)
        self.fcb3_1 = nn.Linear(128,64)
        self.fcb3_2 = nn.Linear(64,32)
        self.fcb3_3 = nn.Linear(32,8)
        self.fcb4_1 = nn.Linear(8,1)
        
        # Dropout 
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.2)
        
        # Batchnorm
        self.bn1 = nn.BatchNorm1d(256)

        # Full range output for hubYX location
        self.ifFullRange = ifFullRange
        
        if ifFullRange: 
            print("Model with full HubYX loc")
        else: 
            print("Model with nomalized HubYX loc")

        
        
    
    def forward(self,x): 

        x1 = self.resnet18(x)

        # Regressor A
        xa1 = torch.flatten(x1,1)
        xa1 = self.dropout(xa1)
        
        xa2 = F.relu(self.bn1(self.fca1(xa1)))
        #xa2 = self.dropout(xa2)
        
        if self.ifFullRange: 
            xa5 = self.fca2(xa2)
        else: 
            xa5 = F.sigmoid(self.fca2(xa2))

        # Regressor B
        xb2 = F.relu(self.fcb1(xa1))
        xb2 = self.dropout2(xb2)
        
        xb3 = F.relu(self.fcb2(xb2))
        xb3 = self.dropout2(xb3)
        
        xb4 = F.relu(self.fcb3_1(xb3))
        xb4 = self.dropout2(xb4)
        
        xb4 = F.relu(self.fcb3_2(xb4))
        xb4 = self.dropout2(xb4)
        
        xb4 = F.relu(self.fcb3_3(xb4))
        xb4 = self.dropout2(xb4)
        
        xb5 = F.sigmoid(self.fcb4_1(xb4))
        
        return xa5, xb5
    
class ResnetEncoderRegressorHubTIp(nn.Module):
    
    def __init__(self, ifFullRange = False): 
        super(ResnetEncoderRegressorHubTIp, self).__init__()
        
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # model = models.resnet18()
        self.resnet18 = torch.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
        
        # Regressor A  # 57344
        self.fca1 = nn.Linear(32768,512)
        self.fca2 = nn.Linear(512,256)
        self.fca3 = nn.Linear(256,32)
        self.fca4 = nn.Linear(32,4)
        self.fca5 = nn.Linear(256,32)

        # Regressor B
        self.fcb1 = nn.Linear(512,256)
        self.fcb2 = nn.Linear(256,128)
        self.fcb3_1 = nn.Linear(128,64)
        self.fcb3_2 = nn.Linear(64,32)
        self.fcb3_3 = nn.Linear(32,8)
        self.fcb4_1 = nn.Linear(8,1)
        
        # Dropout 
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.2)
        
        
        # Full range output for hubYX location
        self.ifFullRange = ifFullRange
        
        if ifFullRange: 
            print("Model with full HubYX loc")
        else: 
            print("Model with nomalized HubYX loc")

        
        
    
    def forward(self,x): 

        # Encoder 
        # x1 = F.relu(self.conv1(x)) #256x3 -> 256x16
        # x2 = self.pool(x1) # 256x16 -> 128x16
        # x3 = F.relu(self.conv2(x2)) # 128x16 -> 128x32
        # x4 = self.pool(x3) # 128x32 -> 64 x 32
        # x5 = F.relu(self.conv3(x4)) # 64x32 -> 64x64
        # x6 = self.pool(x5) # 64x64 -> 32x64 
        # x7 = F.relu(self.conv4(x6)) # 32x64 -> 32x128
        # x7 = self.pool(x7)

        x7 = self.resnet18(x)

        # Regressor A
        xa1 = torch.flatten(x7,1)
        xa1 = self.dropout(xa1)
        
        #xa2 = F.relu(self.fca1(xa1))
        #xa2 = self.dropout(xa2)
        
        xa3 = F.relu(self.fca2(xa1))
        xa3 = self.dropout(xa3)
    
        xa4 = F.relu(self.fca3(xa3))
        xa4 = self.dropout(xa4)
    
        if self.ifFullRange: 
            xa5 = self.fca4(xa4)
        else: 
            xa5 = F.sigmoid(self.fca4(xa4))

        # Regressor B
        xb2 = F.relu(self.fcb1(xa1))
        xb2 = self.dropout2(xb2)
        
        xb3 = F.relu(self.fcb2(xb2))
        xb3 = self.dropout2(xb3)
        
        xb4 = F.relu(self.fcb3_1(xb3))
        xb4 = self.dropout2(xb4)
        
        xb4 = F.relu(self.fcb3_2(xb4))
        xb4 = self.dropout2(xb4)
        
        xb4 = F.relu(self.fcb3_3(xb4))
        xb4 = self.dropout2(xb4)
        
        xb5 = F.sigmoid(self.fcb4_1(xb4))
        
        return xa5, xb5