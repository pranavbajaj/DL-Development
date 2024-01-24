"""
Custom loss functions for hub regression

__author__ = "P Rajan"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LineDistanceLoss(nn.Module):
    def __init__(self):
        super(LineDistanceLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # predictions/targets = [y1 x1 y2 x2]
        #
        # Ax + By + C = 0 (Line eq)
        # (y1 - y2)x + (x2 - x1)y + x1*y2 - x2*y1 (Line eq)
        A = targets[:, 0] - targets[:, 2] # A = y1 - y2
        B = targets[:, 3] - targets[:, 1] # B = x2 - x1
        C = torch.mul(targets[:, 1] , targets[:, 2] ) - torch.mul(targets[:, 3] , targets[:, 0] ) # C = x1*y2 - x2*y1
        D = torch.abs(torch.mul(A, predictions[:, 1]) + torch.mul(B, predictions[:, 0]) + C)/ (torch.sqrt(torch.mul(A,A) + torch.mul(B,B)) + 1e-10)
        return torch.mean(D)
    
# class MSELossHubTip(nn.Module):
#     def __init__(self):
#         super(LineDistanceLoss, self).__init__()
    
#     def forward(self, predictions, targets):
#         # predictions/targets = [y1 x1 y2 x2]
#         # Ax + By + C = 0 (Line eq)
#         # (y1 - y2)x + (x2 - x1)y + x1*y2 - x2*y1 (Line eq)
#         A = targets[:, 0] - targets[:, 2] # A = y1 - y2
#         B = targets[:, 3] - targets[:, 1] # B = x2 - x1
#         C = torch.mul(targets[:, 1] , targets[:, 2] ) - torch.mul(targets[:, 3] , targets[:, 0] ) # C = x1*y2 - x2*y1
#         D = torch.abs(torch.mul(A, predictions[:, 1]) + torch.mul(B, predictions[:, 0]) + C)/ (torch.sqrt(torch.mul(A,A) + torch.mul(B,B)) + 1e-10)
#         return torch.mean(D)
    
# class LineAngleLoss(nn.Module): 
#     def __init__(self):
#         super(LineDistanceLoss, self).__init__()
    
#     def forward(self, predictions, targets):
#         # predictions = [(x1 y1), (x2 y2)]
#         diff_hub_tip = targets[0:1] - targets[2:3] 
#         diff_predhub_tip = predictions[0:1] - predictions[2:3] 
#         torch.nn.CosineSimilarity(diff_hub_tip, diff_predhub_tip)
#         return torch.acos((predictions - targets) ** 2)

