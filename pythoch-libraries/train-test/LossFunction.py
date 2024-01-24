import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
              
        IoULoss = 0.0         
        
        for i in range(len(inputs)): 
            
            inp = inputs[i].view(-1)
            tar = targets[i].view(-1)

            #intersection is equivalent to True Positive count
            #union is the mutually inclusive area of all labels & predictions 
            intersection = (inp * tar).sum()
            total = (inp + tar).sum()
            union = total - intersection 

            IoU = (intersection + smooth)/(union + smooth)
            IoULoss += (1 - IoU)
        
        IoUloss = IoULoss / len(inputs)
                
        return IoUloss
    
    

    
def MSE(inp, tar):
        
    return torch.mean((((inp[0] - tar[0]) ** 2) + ((inp[1] - tar[1]) ** 2)) ** 0.5 )


# def MSE(inp, tar):
        
#     return (((torch.argmax(inp[0]) - tar[0]) ** 2) + ((torch.argmax(inp[1]) - tar[1]) ** 2)) ** 0.5 