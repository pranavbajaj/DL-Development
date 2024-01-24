"""

Custom Autoencoders 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F



### Convolution Blocks ### 

# Convolution Block without any activation 
class ConvBlockBase(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size): 
        super(ConvBlockBase, self).__init__()
       
        self.conv_block = nn.Sequential(nn.Conv2d(input_ch, output_ch, kernel_size, padding = int(kernel_size / 2)),
                                        nn.BatchNorm2d(output_ch))
        
    def forward(self, x):
        return self.conv_block(x)


# Convolution Block with activation 
class ConvBlockReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size): 
        super(ConvBlockReLU, self).__init__()
       
        self.conv_block = nn.Sequential(ConvBlockBase(input_ch, output_ch, kernel_size), 
                                        nn.ReLU())
        
    def forward(self, x):
        return self.conv_block(x)

### Convolution Blocks End ### 
    
    
    
### Transpose Convolution Blocks ###    
    
# Transpose Convolution Block without any activation     
class TConvBlockBase(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(TConvBlockBase, self).__init__()
        
        self.tconv_block = nn.Sequential(nn.ConvTranspose2d(input_ch,output_ch,2,stride=2),
                                        nn.BatchNorm2d(output_ch))
        
    def forward(self, x):
        return self.tconv_block(x)

# Transpose Convolution Block with ReLU activation 
class TConvBlockReLU(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(TConvBlockReLU, self).__init__()
        
        self.tconv_block = nn.Sequential(TConvBlockBase(input_ch, output_ch),
                                        nn.ReLU())
        
    def forward(self, x):
        return self.tconv_block(x)

### Transpose Convolution Block End ###    
    
    
    
### Inception Blocks ###    
    
# Inception Block without any final activation, but ReLU as intermediate activations 
class InceptionBlockBase(nn.Module):        
    def __init__(self, input_ch, cha_1x1, cha_1x1_3x3, cha_1x1_5x5, cha_3x3_1x1): 
        super(InceptionBlockBase, self).__init__()
        
        self.group1 = ConvBlockBase(input_ch, cha_1x1, 1)
        self.group2 = nn.Sequential(ConvBlockReLU(input_ch, cha_1x1_3x3[0], 1),
                                   ConvBlockBase(cha_1x1_3x3[0],cha_1x1_3x3[1],3))
        self.group3 = nn.Sequential(ConvBlockReLU(input_ch, cha_1x1_5x5[0], 1),
                                   ConvBlockBase(cha_1x1_5x5[0],cha_1x1_5x5[1],5))
        self.group4 = nn.Sequential(ConvBlockReLU(input_ch, cha_3x3_1x1[0], 3),
                                   ConvBlockBase(cha_3x3_1x1[0],cha_3x3_1x1[1],1))
        
    def forward(self, x):    
        g1 = self.group1(x)
        g2 = self.group2(x)
        g3 = self.group3(x)
        g4 = self.group4(x)
        return torch.cat((g1, g2, g3, g4), dim=1)

    
# Inception Block with ReLU activation    
class InceptionBlockReLU(nn.Module):
    def __init__(self, input_ch, cha_1x1, cha_1x1_3x3, cha_1x1_5x5, cha_3x3_1x1): 
        super(InceptionBlockReLU, self).__init__()
        
        self.layer = nn.Sequential(InceptionBlockBase(input_ch, cha_1x1, cha_1x1_3x3, cha_1x1_5x5, cha_3x3_1x1),
                                  nn.ReLU())
        
    def forward(self, x):    
        return self.layer(x)

### Inception Blocks End ###     
    
    
### Residual Blocks ###

# Residual Block without any activation    
class ResidualBlockBase(nn.Module):
    def __init__(self, input_ch, kernel_size):
        super(ResidualBlockBase, self).__init__()
        
        self.residual_layer = nn.Sequential(nn.Conv2d(input_ch, input_ch, kernel_size, padding = int(kernel_size / 2)),
                                           nn.BatchNorm2d(input_ch))
        
    def forward(self,x):
        x1 = self.residual_layer(x)
        return x + x1

    
# Residual Block with ReLU activation     
class ResidualBlockReLU(nn.Module):
    def __init__(self, input_ch, kernel_size):
        super(ResidualBlockReLU, self).__init__()
        
        self.residual_layer = nn.Sequential(ResidualBlockBase(input_ch,kernel_size),
                                            nn.ReLU())
        
    def forward(self,x):
        return self.residual_layer(x)
    
    
### Residual Block End ###


### Residual Linear Block base ###
class ResidualLinearBlockBase(nn.Module):
    def __init__(self,input_nodes):
        super(ResidualLinearBlockBase, self).__init__()
        
        self.linear_layer = nn.Sequential(nn.Linear(input_nodes,input_nodes),
                                         nn.BatchNorm1d(input_nodes))
        
        
    def forward(self,x):
        return self.linear_layer(x)
    
### Residual Linear Block base End ###


### Residual Linear Block ReLU Activate ###
class ResidualLinearBlockReLU(nn.Module):
    def __init__(self,input_nodes):
        super(ResidualLinearBlockReLU, self).__init__()
        
        self.linear_layer = nn.Sequential(ResidualLinearBlockBase(input_nodes),
                                         nn.ReLU())
        
    def forward(self,x):
        return self.linear_layer(x)
    
### Residual Linear Block ReLU Activate End ###



    
    
    
    
    