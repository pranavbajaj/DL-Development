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


def Test_Shaft_Segmentation(model_name = "test.pt",
		parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena",
		test_data_folder = "test_data",
		sub_folders = ["EL"], 
		img_dir = "images_cropped", 
		mask_dir = "masks_cropped", 
		csv_filename = "cropped.csv", 
		output_dir="D:/test_dir",
		threshold = 0.5):


		output_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/results/" + output_dir 
		model = torch.jit.load(parent_dir + "/models/shaft_segmentation/" + model_name)
		model.to(device)
		model.eval() 

		test_data_name = ""
		for ele in sub_folders:
			test_data_name += (ele + ",")

		dfs = []
		for ele in sub_folders: 
			df = pd.read_csv(parent_dir + "/" + test_data_folder + "/" + ele +"/" + csv_filename)
			dfs.append(df)

		images_path = []
		masks_path = []
		imgMask_names = []

		for i in range(len(dfs)): 
			for name in dfs[i]["Images"]: 
				images_path.append(parent_dir + "/" + test_data_folder + "/" + sub_folders[i] + "/" + img_dir + "/" + name)
				masks_path.append(parent_dir + "/" + test_data_folder + "/" + sub_folders[i] + "/" + mask_dir + "/" + name)
				imgMask_names.append(name)

		IoU_losses = []
		loss = IoULoss() 

		angle_losses = []

		createFolder(output_path + "/Overlap_Img_PredMask_RGB")
		createFolder(output_path + "/Predicted_Masks_Gray")
		#createFolder(output_path + "/Overlap_PredTrue_Mask_RGB")
		#createFolder(output_path + "/merged")

		fps = []


		for n in range(len(images_path)):
			
			# rgb_input_tensor is original RGB image in tensor format with uint8 format  
			# input_tensor is RGB image scaled to 0-1 values in device format. 
			rgb_input_tensor = read_image(images_path[n],mode=ImageReadMode.RGB)
			input_tensor = rgb_input_tensor.type(torch.FloatTensor)
			input_tensor = input_tensor / 255
			input_tensor = input_tensor.to(device)

			# original_label is true label in tensor with float64 formate 
			# true_label_tensor is tensor with flaot64 in device format 
			original_label = read_image(masks_path[n])
			original_label = original_label.type(torch.FloatTensor)
			true_label_tensor = original_label.to(device)
			_, height_, width_ = true_label_tensor.size() 
			# Prdiction 
			with torch.no_grad(): 
				start_time = time.time() 
				predicted_label_tensor_without_threshold = model.forward(torch.reshape(input_tensor,(1,3,height_,width_)))
				ifps = 1 / ((time.time() - start_time) + 0.00001)
				fps.append(ifps)
				print("Fequency: ", ifps)

			# Applying threshold to the output data. 
			thr_object = torch.nn.Threshold(threshold, 0.0)
			predicted_label_tensor = thr_object(predicted_label_tensor_without_threshold[0])
					
			# Appending the loss 
			l = loss.forward(true_label_tensor, predicted_label_tensor)
			l = l.cpu().numpy() # Check if l.item() can replace this 
			IoU_losses.append((1 - l))

			# Converting predicted label to numpy array 
			predicted_label_np = np.transpose(predicted_label_tensor.cpu().detach().numpy(), (1,2,0))

			# Converting true label to numpy array 
			true_label_np = np.transpose(original_label.numpy(), (1,2,0))

			# Line fit on ture label 
			h, w, _ = np.shape(true_label_np)
			reg_true = LineFitOnMask(true_label_np[:,:,0],threshold)
			true_linefit_mask = LineMaskUint(reg_true, height= h, width=w,color="g")
            
            
			# Line fit on predicted label 
			h, w, _ = np.shape(predicted_label_np)
			reg_predicted = LineFitOnMask(predicted_label_np[:,:,0],threshold)
			if reg_predicted: 
				predicted_linefit_mask = LineMaskUint(reg_predicted, height= h, width=w,color="r")
            
    
    
			# 2 Line angle error (degree)
			if reg_predicted: 
				m1 = reg_true.estimator_.coef_
				m2 = reg_predicted.estimator_.coef_
				angle = abs((m2 - m1)/(1 + m1*m2))
				ret = atan(angle)
				de = (ret * 180) / np.pi
				angle_losses.append(de)
			else: 
				print("No needle detected for image: ", imgMask_names[n])

			# Saving output. 
			## Gray Mask 
			cv2.imwrite(output_path + "/Predicted_Masks_Gray/" + imgMask_names[n], predicted_label_np)

			## Merging ture and predictd line 
			if reg_predicted:
				merge_lines = cv2.add(true_linefit_mask, predicted_linefit_mask)
			else: 
				merge_lines = true_linefit_mask

			merge_lines = cv2.cvtColor(merge_lines,cv2.COLOR_BGR2RGB)
			#cv2.imwrite(output_path + "/Overlap_PredTrue_Mask_RGB/" + imgMask_names[n], merge_lines)

			## Merging predicted mask and original image 
			rgb_input_np = rgb_input_tensor.numpy() 
			rgb_input_np = np.transpose(rgb_input_np, (1,2,0))
			
			masked_b = np.zeros([height_,width_], dtype=np.uint8)
			masked_g = np.zeros([height_,width_], dtype=np.uint8)
			masked_r = np.zeros([height_,width_], dtype=np.uint8)

			for i in range(len(predicted_label_np)): 
				for j in range(len(predicted_label_np[0])): 
					if predicted_label_np[i][j] > threshold: 
						masked_r[i][j] = 255
						masked_r[i][j] = 255


			merged_img = cv2.merge((masked_r, masked_g, masked_b))

			overlap_img = cv2.addWeighted(rgb_input_np, 0.5, merged_img, 0.5, 0)
			overlap_img = cv2.cvtColor(overlap_img,cv2.COLOR_BGR2RGB)

			cv2.imwrite(output_path + "/Overlap_Img_PredMask_RGB/" + imgMask_names[n], overlap_img)
			
			final_merge = cv2.vconcat((overlap_img, merge_lines))

			#cv2.imwrite(output_path + "/merged/" + imgMask_names[n], final_merge)
		
		print("Mean FPS: ", np.mean(fps[2:]))

		field_names = ['Date', 'Test Data', 'Output Dir', 'Model Name', 'IoU', 'IoU std', 'IoU min', 'IoU max', 'Deg error', 
					'Deg std', 'Deg min', 'Deg max', 'Threshold']

		t_ = datetime.datetime.now()
		date = str(t_.month) + "/" + str(t_.day) + "/" + str(t_.year)
		iou_loss = np.mean(IoU_losses)  
		iou_std = np.std(IoU_losses)
		iou_min = np.min(IoU_losses)
		iou_max = np.max(IoU_losses)
		deg_error = np.mean(angle_losses)
		deg_std = np.std(angle_losses)
		deg_min = np.min(angle_losses)
		deg_max = np.max(angle_losses)

		dict = {'Date':date, 'Test Data':test_data_name, 'Output Dir':output_dir, 'Model Name': model_name, 
					'IoU':iou_loss, 'IoU std':iou_std, 'IoU min':iou_min, 'IoU max':iou_max, 'Deg error':deg_error, 
					'Deg std':deg_std, 'Deg min':deg_min, 'Deg max':deg_max, 'Threshold':threshold}

		#result_csv = "//cgmqnap.clearguide.local/data/Needles/Lumena/result/result.csv"
		result_csv = parent_dir + "/results/result.csv"

		with open(result_csv, 'a') as f_object:
			dictwriter_object = DictWriter(f_object, fieldnames=field_names)
			dictwriter_object.writerow(dict)
			f_object.close()



if __name__ == "__main__": 
	
	print("Test")
    
	model_name = "model_cyc_3_ep_200_bs_128_lr_0.0001_20240122_T192512finetuned.pt"
	output_dir = "TT_output"
	parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena/"
	test_data_folder = "test_data"



	output_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/test_data/SS_3/" + output_dir
	input_path = "//cgmqnap.clearguide.local/data/Needles/Lumena/test_data/SS_3/images/"


# 	sub_folders = ["PR", "PC2", "EL_2", "EL", "CNMC"]
	sub_folders = ["PR", "PC2", "EL_2", "EL", "CNMC"]
	csv_filename = [ "cropped_256x256.csv"]
	img_dir = [ "images"]
	mask_dir = [ "masks"]   

# 	csv_filename = "cropped_256x256_original.csv"
# 	img_dir = "images_cropped_256x256_original"
# 	mask_dir = "masks_cropped_256x256_original"

	threshold = 0.7
	FullRange = False 
	ifAddMskCha = False 
	ifOnlyRegA = False 

	Test_Shaft_Segmentation(model_name = model_name, parent_dir = parent_dir, test_data_folder = test_data_folder, img_dir = img_dir[0], mask_dir = mask_dir[0], csv_filename = csv_filename[0], output_dir = output_dir, threshold = threshold, sub_folders = sub_folders)

    

    
