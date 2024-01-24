import cv2
import os 
import time 
import pandas as pd
import numpy as np
import datetime
from argparse import ArgumentParser
from math import atan
from csv import DictWriter
import sys 
sys.path.append(os.path.join('..','segment-utils'))
from VideoAnnotationHelper import LineFitOnMask, LineMaskUint,LineMaskfloat32, SaveImage, createFolder
from LossFunction import IoULoss, MSE
import glob 
import torch
import torch.nn as nn
import CustomImageDataset as cid
from torchvision.io import read_image, ImageReadMode

device = ("cuda" if torch.cuda.is_available() else "cpu")


def TestFolder(model_name = "",
				parent_dir ="", 
				input_path ="",
				input_type = ".png", 
				output_path = "",
				threshold = 0.7):

	createFolder(output_path)

	image_paths =  glob.glob(input_path + "/*" + input_type)
	test_dataset = cid.CustomImageGrayMaskDataset_dataAugmentation(image_paths, None)

	model = torch.jit.load(parent_dir + "/models/shaft_segmentation/" + model_name)
	model.to(device)
	model.eval() 

	i = 0 
	for image in test_dataset: 
		
		_, height_, width_ = image.size() 
		image_gpu = image.to(device)

		with torch.no_grad():
			predicted_label_tensor = model.forward(torch.reshape(image_gpu,(1,3,height_,width_)))
		


		thr_object = torch.nn.Threshold(threshold, 0.0)
		predicted_label_tonsor = thr_object(predicted_label_tensor[0])

		predicted_label_np = np.transpose(predicted_label_tonsor.cpu().detach().numpy(), (1,2,0))
		input_np = np.transpose(image.numpy(), (1,2,0))

		masked_r = np.zeros([height_,width_,1],np.float32)
		merged_img = cv2.merge((predicted_label_np, masked_r, predicted_label_np))
		overlap_img = cv2.addWeighted(input_np * 255, 0.5, merged_img * 255, 0.5, 0)

		final_output = cv2.cvtColor(overlap_img, cv2.COLOR_BGR2RGB)

		cv2.imwrite(output_path + "/output" +str(i)+ ".png" , final_output)
		i += 1 


if __name__ == "__main__": 
	
	print("Testing Needle Segmentation")


	model_name = "model_cyc_3_ep_200_bs_128_lr_0.001_20240116_T102132.pt"
	parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena"
	input_path = parent_dir + "/test_data/CGM02.Shoulder/images"
	output_path = parent_dir + "/results/" + "CC"

	TestFolder(model_name = model_name, 
				parent_dir = parent_dir,
				input_path = input_path,
				output_path= output_path)