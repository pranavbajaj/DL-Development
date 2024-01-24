"""

Custom Autoencoders 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomModelHelper import * 
    
    
class HubRegression1(nn.Module):
    
    def __init__(self):
        super(HubRegression1, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(4,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ConvBlockReLU(32,32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[16,16],[16,16],[16,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[32,32],[32,32],[32,32]),
                                  ResidualBlockReLU(128,3))  
        
        self.pool = nn.MaxPool2d(2,2)
        
        
        # Regressor A  # 57344
        
        self.fca = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                )
        
        self.fca1 = nn.Sequential(nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        self.fca2 = nn.Sequential(nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # FCN 
        x8_f = torch.flatten(x8,1)
        x9 = self.fca(x8_f)
        
        # Output
        
        x10 = self.fca1(x9)
        x11 = self.fca2(x9)
        
        return torch.cat((x10,x11), -1)
    
    
class HubRegression2(nn.Module):
    
    def __init__(self):
        super(HubRegression2, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(4,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ConvBlockReLU(32,32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[16,16],[16,16],[16,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[32,32],[32,32],[32,32]),
                                  ResidualBlockReLU(128,3))  
        
        self.pool = nn.MaxPool2d(2,2)
        
        
        # Regressor A  # 57344
        
        self.fca = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                )
        
        self.fca1 = nn.Sequential(nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        self.fca2 = nn.Sequential(nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        self.fcb = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # FCN 
        x8_f = torch.flatten(x8,1)
        x9 = self.fca(x8_f)
        
        # Output
        
        x10 = self.fca1(x9)
        x11 = self.fca2(x9)
        
        x12 = self.fcb(x8_f)
        
        return torch.cat((x10,x11), -1), x12 


class TipHubRegression(nn.Module):
    
    def __init__(self):
        super(TipHubRegression, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(3,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ConvBlockReLU(32,32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[16,16],[16,16],[16,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4_h = nn.Sequential(InceptionBlockReLU(64,32,[32,32],[32,32],[32,32]),
                                  ResidualBlockReLU(128,3)) 

        self.conv4_t = nn.Sequential(InceptionBlockReLU(64,32,[32,32],[32,32],[32,32]),
                                  ResidualBlockReLU(128,3))
        
        self.pool = nn.MaxPool2d(2,2)
        
        
        # Regressor A  # 57344
        
        self.fcah = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,2),
                                nn.Sigmoid()
                                )
        

        self.fcat = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,2),
                                nn.Sigmoid()
                                )
        
        
        self.fcb = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7_h = self.conv4_h(x6) # 128x32
        x8_h = self.pool(x7_h) # 128x16 

        x7_t = self.conv4_t(x6) # 128x32
        x8_t = self.pool(x7_t) # 128x16 
        
        # FCN hub
        x8_hf = torch.flatten(x8_h,1)
        x9_h = self.fcah(x8_hf)

        # FCN tip
        x8_tf = torch.flatten(x8_t,1)
        x9_t = self.fcat(x8_tf)

        # Reg B
        x12 = self.fcb(x8_hf)
        
        return torch.cat((x9_h, x9_t), -1), x12 


class TipHubRegression2(nn.Module):
    
    def __init__(self):
        super(TipHubRegression2, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(4,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ResidualBlockReLU(32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[16,16],[16,16],[16,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4 = nn.Sequential(ConvBlockReLU(64,128,3)) 

        
        self.pool = nn.MaxPool2d(2,2)
        
        
        # Regressor A  # 57344
        
        self.fca = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                )

        self.fcah = nn.Sequential(nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,2),
                                nn.Sigmoid())
        

        self.fcat = nn.Sequential(nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,2),
                                nn.Sigmoid()
                                )
        
        
        self.fcb = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 

        # FCN 
        x8_f = torch.flatten(x8,1)
        x9 = self.fca(x8_f)
        
        # FCN hub
        x9_h = self.fcah(x9)

        # FCN tip
        x9_t = self.fcat(x9)

        # Reg B
        x12 = self.fcb(x8_f)
        
        return torch.cat((x9_h, x9_t), -1), x12 

class TipHubRegression3(nn.Module):
    
    def __init__(self):
        super(TipHubRegression3, self).__init__()
        
        self.conv1 = nn.Sequential(InceptionBlockReLU(4,4,[8,4],[8,4],[8,4]))
        
        self.conv2 = nn.Sequential(InceptionBlockReLU(16,8,[16,8],[16,8],[16,8]))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[16,16],[16,16],[16,16]))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[32,32],[32,32],[32,32])) 

        
        self.pool = nn.MaxPool2d(2,2)
        
        
        # Regressor A  # 57344
        
        self.fca = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                )

        self.fcah = nn.Sequential(nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,2),
                                nn.Sigmoid())
        

        self.fcat = nn.Sequential(nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,2),
                                nn.Sigmoid())
        
        
        self.fcb = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 

        # FCN 
        x8_f = torch.flatten(x8,1)
        x9 = self.fca(x8_f)
        
        # FCN hub
        x9_h = self.fcah(x9)

        # FCN tip
        x9_t = self.fcat(x9)

        # Reg B
        x12 = self.fcb(x8_f)
        
        return torch.cat((x9_h, x9_t), -1), x12 
    
    
class TipHubRegression4(nn.Module):
    
    def __init__(self):
        super(TipHubRegression4, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(4,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ConvBlockReLU(32,32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[16,16],[16,16],[16,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[32,32],[32,32],[32,32]),
                                  ResidualBlockReLU(128,3)) 

        
        
        self.pool = nn.MaxPool2d(2,2)
        
        
        # Regressor A  # 57344
        
        self.fca = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,4),
                                nn.Sigmoid()
                                )
        
        
        self.fcb = nn.Sequential(nn.Linear(32768,512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256,32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Linear(32,1),
                                nn.Sigmoid())
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # FCN hub
        x8_f = torch.flatten(x8,1)
        x9 = self.fca(x8_f)

        # Reg B
        x12 = self.fcb(x8_hf)
        
        return x9, x12 
    
class HubRegression(nn.Module):
    
    def __init__(self):
        super(HubRegression, self).__init__()
               
        self.conv1 = nn.Sequential(ConvBlockReLU(4,16,7),
                                  ConvBlockReLU(16,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ConvBlockReLU(32,32,5),
                                  ConvBlockReLU(32,32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[32,16],[32,16],[32,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[64,32],[64,32],[64,32]),
                                  ResidualBlockReLU(128,3))
        
        self.conv5 = nn.Sequential(nn.Conv2d(32, 32, 3, padding = 1, stride = 2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(), 
                                   nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Flatten(), 
                                   nn.Linear(32,1),
                                   nn.Sigmoid())
        
        self.conv6 = nn.Sequential(nn.Conv2d(32, 32, 3, padding = 1, stride = 2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(), 
                                   nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Flatten(), 
                                   nn.Linear(32,1),
                                   nn.Sigmoid())
        
        self.pool = nn.MaxPool2d(2,2)
       
        
        
        self.t_conv1 = nn.Sequential(TConvBlockReLU(128,64),
                                    ResidualBlockBase(64,3)) 

        self.t_conv2 = nn.Sequential(TConvBlockReLU(64,32),
                                    ResidualBlockBase(32,3))
        
        
        self.jmp_conn3 = nn.Conv2d(64, 32, 1, padding = 0)
        self.jmp_conn4 = nn.Conv2d(128, 64, 1, padding = 0)
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # Decoder 
        x9 = self.t_conv1(x8) # 64x32
        x10 = F.relu(x9 + self.jmp_conn4(x7)) # 64x32
        
        x11 = self.t_conv2(x10) # 32x64
        x12 = F.relu(x11 + self.jmp_conn3(x5)) # 32x64
        
        o1 = self.conv5(x12) 
        o2 = self.conv6(x12) 
        
        return torch.cat((o1,o2),-1)
        
        
    
class NeedleSegmentationModel(nn.Module):
    
    def __init__(self):
        super(NeedleSegmentationModel, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(3,16,7),
                                  ConvBlockReLU(16,16,7),
                                  ConvBlockReLU(16,16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ConvBlockReLU(32,32,5),
                                  ConvBlockReLU(32,32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[32,16],[32,16],[32,16]),
                                  ResidualBlockReLU(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[64,32],[64,32],[64,32]),
                                  ResidualBlockReLU(128,3))
        
        self.conv5 = nn.Sequential(nn.Conv2d(4, 1, 3, padding = 1),
                                  nn.Sigmoid())
        
        self.pool = nn.MaxPool2d(2,2)
        
        
        self.t_conv1 = nn.Sequential(TConvBlockReLU(128,64),
                                    ResidualBlockBase(64,3)) 

        self.t_conv2 = nn.Sequential(TConvBlockReLU(64,32),
                                    ResidualBlockBase(32,3))
        
        self.t_conv3 = nn.Sequential(TConvBlockReLU(32,16),
                                    ResidualBlockBase(16,3))
        
        self.t_conv4 = nn.Sequential(TConvBlockReLU(16,4),
                                    ResidualBlockReLU(4,3))
        
        self.jmp_conn2 = nn.Conv2d(32, 16, 1, padding = 0)
        self.jmp_conn3 = nn.Conv2d(64, 32, 1, padding = 0)
        self.jmp_conn4 = nn.Conv2d(128, 64, 1, padding = 0)
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # Decoder 
        x9 = self.t_conv1(x8) # 64x32
        x10 = F.relu(x9 + self.jmp_conn4(x7)) # 64x32
        
        x11 = self.t_conv2(x10) # 32x64
        x12 = F.relu(x11 + self.jmp_conn3(x5)) # 32x64
        
        x13 = self.t_conv3(x12) # 16x128
        x14 = F.relu(x13 + self.jmp_conn2(x3)) # 16x128
        
        x15 = self.t_conv4(x14)
        x16 = self.conv5(x15)
        
        return x16
    
    
        
class NeedleSegmentationModel_1(nn.Module):
    
    def __init__(self):
        super(NeedleSegmentationModel_1, self).__init__()
        self.conv1 = nn.Sequential(ConvBlockReLU(3,16,7),
                                  ResidualBlockBase(16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ResidualBlockBase(32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[4,16],[4,16],[4,16]),
                                  ResidualBlockBase(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[8,32],[8,32],[8,32]),
                                  ResidualBlockBase(128,3))
        
        
        
        self.conv5 = nn.Sequential(nn.Conv2d(20, 1, 3, padding = 1),
                                  nn.Sigmoid())
        
        self.conv5_hub = nn.Sequential(nn.Conv2d(20, 1, 3, padding = 1),
                                  nn.Sigmoid())
        
#         self.conv5_tip = nn.Sequential(nn.Conv2d(20, 1, 3, padding = 1),
#                                   nn.Sigmoid())
        
        self.pool = nn.MaxPool2d(2,2)
        
        # Shaft 
        self.t_conv1 = nn.Sequential(TConvBlockReLU(128,64)) 
        self.t_conv2 = nn.Sequential(TConvBlockBase(64,32))
        self.t_conv3 = nn.Sequential(TConvBlockBase(32,16))
        self.t_conv4 = nn.Sequential(TConvBlockBase(16,4))
        self.jmp_conn3 = nn.Conv2d(32, 16, 1, padding = 0)
        self.jmp_conn4 = nn.Conv2d(64, 32, 1, padding = 0)
        self.jmp_conn5 = nn.Conv2d(128, 64, 1, padding = 0)
        
        # Hub 
        self.t_conv1_hub = nn.Sequential(TConvBlockReLU(128,64)) 
        self.t_conv2_hub = nn.Sequential(TConvBlockBase(64,32))
        self.t_conv3_hub = nn.Sequential(TConvBlockBase(32,16))
        self.t_conv4_hub = nn.Sequential(TConvBlockBase(16,4))
        self.jmp_conn3_hub = nn.Conv2d(32, 16, 1, padding = 0)
        self.jmp_conn4_hub = nn.Conv2d(64, 32, 1, padding = 0)
        self.jmp_conn5_hub = nn.Conv2d(128, 64, 1, padding = 0)
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # Decoder A
        x9 = self.t_conv1(x8) # 64x32
        X10 = F.relu(x9 + self.jmp_conn5(x7))
        
        x11 = self.t_conv2(X10) # 32x64
        x12 = F.relu(x11 + self.jmp_conn4(x5)) # 32x64

        x13 = self.t_conv3(x12) # 16x128
        x14 = F.relu(x13 + self.jmp_conn3(x3)) # 48x128
        
        x15 = self.t_conv4(x14) # 4x256
        x16 = torch.cat((x15, x1),dim = 1)
        
        x17 = self.conv5(x16) # 1x256
        
        # Decoder Hub 
        
        x9_hub = self.t_conv1_hub(x8) # 64x32
        X10_hub = F.relu(x9_hub + self.jmp_conn5_hub(x7))
        
        x11_hub = self.t_conv2_hub(X10_hub) # 32x64
        x12_hub = F.relu(x11_hub + self.jmp_conn4_hub(x5)) # 32x64

        x13_hub = self.t_conv3_hub(x12_hub) # 16x128
        x14_hub = F.relu(x13_hub + self.jmp_conn3_hub(x3)) # 48x128
        
        x15_hub = self.t_conv4_hub(x14_hub) # 4x256
        x16_hub = torch.cat((x15_hub, x1),dim = 1)
        
        x17_hub = self.conv5_hub(x16_hub) # 1x256
        
        
        return x17, x17_hub 

class NeedleSegmentationModel_Shaft(nn.Module):
    
    def __init__(self):
        super(NeedleSegmentationModel_Shaft, self).__init__()
        
        self.conv1 = nn.Sequential(ConvBlockReLU(3,16,7),
                                  ResidualBlockBase(16,7))
        
        self.conv2 = nn.Sequential(ConvBlockReLU(16,32,5),
                                  ResidualBlockBase(32,5))
        
        self.conv3 = nn.Sequential(InceptionBlockReLU(32,16,[4,16],[4,16],[4,16]),
                                  ResidualBlockBase(64,3))
        
        self.conv4 = nn.Sequential(InceptionBlockReLU(64,32,[8,32],[8,32],[8,32]),
                                  ResidualBlockBase(128,3))
        
        
        
        self.conv5 = nn.Sequential(nn.Conv2d(20, 1, 3, padding = 1),
                                  nn.Sigmoid())
        
        self.conv5_hub = nn.Sequential(nn.Conv2d(20, 1, 3, padding = 1),
                                  nn.Sigmoid())
        
        self.pool = nn.MaxPool2d(2,2)
        
        # Shaft 
        self.t_conv1 = nn.Sequential(TConvBlockReLU(128,64)) 
        self.t_conv2 = nn.Sequential(TConvBlockBase(64,32))
        self.t_conv3 = nn.Sequential(TConvBlockBase(32,16))
        self.t_conv4 = nn.Sequential(TConvBlockBase(16,4))
        self.jmp_conn3 = nn.Conv2d(32, 16, 1, padding = 0)
        self.jmp_conn4 = nn.Conv2d(64, 32, 1, padding = 0)
        self.jmp_conn5 = nn.Conv2d(128, 64, 1, padding = 0)
        
        self.t_conv1_hub = nn.Sequential(TConvBlockReLU(128,64)) 
        self.t_conv2_hub = nn.Sequential(TConvBlockBase(64,32))
        self.t_conv3_hub = nn.Sequential(TConvBlockBase(32,16))
        self.t_conv4_hub = nn.Sequential(TConvBlockBase(16,4))
        self.jmp_conn3_hub = nn.Conv2d(32, 16, 1, padding = 0)
        self.jmp_conn4_hub = nn.Conv2d(64, 32, 1, padding = 0)
        self.jmp_conn5_hub = nn.Conv2d(128, 64, 1, padding = 0)
        
        
        
        
    def forward(self,x):
        
        # Encoder 
        x1 = self.conv1(x) # 16x256
        x2 = self.pool(x1) # 16x128
        
        x3 = self.conv2(x2) # 32x128
        x4 = self.pool(x3) # 32x64
        
        x5 = self.conv3(x4) # 64x64
        x6 = self.pool(x5) # 64x32
        
        x7 = self.conv4(x6) # 128x32
        x8 = self.pool(x7) # 128x16 
        
        # Decoder A
        x9 = self.t_conv1(x8) # 64x32
        X10 = F.relu(x9 + self.jmp_conn5(x7))
        
        x11 = self.t_conv2(X10) # 32x64
        x12 = F.relu(x11 + self.jmp_conn4(x5)) # 32x64

        x13 = self.t_conv3(x12) # 16x128
        x14 = F.relu(x13 + self.jmp_conn3(x3)) # 48x128
        
        x15 = self.t_conv4(x14) # 4x256
        x16 = torch.cat((x15, x1),dim = 1)
        
        x17 = self.conv5(x16) # 1x256
        
        
        x9_hub = self.t_conv1(x8) # 64x32
        X10_hub = F.relu(x9_hub + self.jmp_conn5(x7))
        
        x11_hub = self.t_conv2(X10_hub) # 32x64
        x12_hub = F.relu(x11_hub + self.jmp_conn4(x5)) # 32x64

        x13_hub = self.t_conv3(x12_hub) # 16x128
        x14_hub = F.relu(x13_hub + self.jmp_conn3(x3)) # 48x128
        
        x15_hub = self.t_conv4(x14_hub) # 4x256
        x16_hub = torch.cat((x15_hub, x1),dim = 1)
        
        x17_hub = self.conv5(x16_hub) # 1x256
        
        return x17


class ConvAutoEncoder(nn.Module):
    
    def __init__(self): 
        super(ConvAutoEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.conv5 = nn.Conv2d(128,256,3,padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.t_conv1 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(256,64,2,stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128,32,2,stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64,16,2,stride=2)
        self.conv6 = nn.Conv2d(32,1,3,padding=1)
        
        self.b1 = nn.BatchNorm2d(16)
        self.b2 = nn.BatchNorm2d(32)
        self.b3 = nn.BatchNorm2d(64)
        self.b4 = nn.BatchNorm2d(128)
        self.b5 = nn.BatchNorm2d(256)
        self.b6 = nn.BatchNorm2d(128)
        self.b7 = nn.BatchNorm2d(64)
        self.b8 = nn.BatchNorm2d(32)
        self.b9 = nn.BatchNorm2d(16)

    
    def forward(self,x): 

        # Encoder 
        x1 = F.relu(self.b1(self.conv1(x))) #256x16
        x2 = self.pool(x1) # 128x16
        x3 = F.relu(self.b2(self.conv2(x2))) # 128x32
        x4 = self.pool(x3) # 64 x 32
        x5 = F.relu(self.b3(self.conv3(x4))) # 64x64
        x6 = self.pool(x5) # 32x64 
        x7 = F.relu(self.b4(self.conv4(x6))) # 32x128
        x8 = self.pool(x7) # 16x128
        x9 = F.relu(self.b5(self.conv5(x8))) # 16 x 256

        #Decoder 
        x10 = F.relu(self.b6(self.t_conv1(x9))) # 32x128
        x11 = torch.cat((x10, x7),dim = 1) # 32x256
        x12 = F.relu(self.b7(self.t_conv2(x11))) # 64x64
        x13 = torch.cat((x12,x5), dim = 1) # 64x128
        x14 = F.relu(self.b8(self.t_conv3(x13))) # 128x32
        x15 = torch.cat((x14,x3), dim = 1) # 128x64
        x16 = F.relu(self.b9(self.t_conv4(x15))) # 256x16
        x17 = torch.cat((x16,x1), dim = 1) # 256x32
        x18 = F.sigmoid(self.conv6(x17)) # 256x1
        
        return x18
        
    
class ConvEncoderRegressor(nn.Module):
    
    def __init__(self, ifFullRange = False): 
        super(ConvEncoderRegressor, self).__init__()
        
        self.conv1 = nn.Conv2d(4,16,3, padding=1,stride = 1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1,stride = 1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1,stride = 1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1,stride = 1)
        self.pool = nn.MaxPool2d(2,2)

        # Regressor A  # 57344
        self.fca1 = nn.Linear(32768,512)
        self.fca2 = nn.Linear(512,256)
        self.fca3 = nn.Linear(256,32)
        self.fca4 = nn.Linear(32,4)

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
        
        # Batch Normalization 
        self.b1 = nn.BatchNorm2d(16)
        self.b2 = nn.BatchNorm2d(32)
        self.b3 = nn.BatchNorm2d(64)
        self.b4 = nn.BatchNorm2d(128)
        
        self.b5 = nn.BatchNorm1d(512)
        self.b6 = nn.BatchNorm1d(256)
        self.b7 = nn.BatchNorm1d(32)

        # Full range output for hubYX location
        self.ifFullRange = ifFullRange
        
        if ifFullRange: 
            print("Model with full HubYX loc")
        else: 
            print("Model with nomalized HubYX loc")
        
    
    def forward(self,x): 

        # Encoder 
        x1 = F.relu(self.b1(self.conv1(x))) #256x3 -> 256x16
        x2 = self.pool(x1) # 256x16 -> 128x16
        x3 = F.relu(self.b2(self.conv2(x2))) # 128x16 -> 128x32
        x4 = self.pool(x3) # 128x32 -> 64 x 32
        x5 = F.relu(self.b3(self.conv3(x4))) # 64x32 -> 64x64
        x6 = self.pool(x5) # 64x64 -> 32x64 
        x7 = F.relu(self.b4(self.conv4(x6))) # 32x64 -> 32x128
        x7 = self.pool(x7)

        # Regressor A
        xa1 = torch.flatten(x7,1)
        xa1 = self.dropout(xa1)
        
        xa2 = F.relu(self.b5(self.fca1(xa1)))
        xa2 = self.dropout(xa2)
        
        xa3 = F.relu(self.b6(self.fca2(xa2)))
        xa3 = self.dropout(xa3)
    
        xa4 = F.relu(self.b7(self.fca3(xa3)))
        xa4 = self.dropout(xa4)
    
        if self.ifFullRange: 
            xa5 = self.fca4(xa4)
        else: 
            xa5 = F.sigmoid(self.fca4(xa4))

        # Regressor B
        xb2 = F.relu(self.fcb1(xa2))
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