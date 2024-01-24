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



def Test_HubRegression(model_name = "test.pt",
	parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena",
	test_data_folder = "test_data",
	sub_folders= ["EL"],
	img_dir = "images_cropped",
	mask_dir = "masks_cropped",
	csv_filename = "cropped.csv", 
	output_dir = "D:/test_dir",
	FullRange = False,
	ifAddMskCha = True,
	ifOnlyRegA = False):


	output_path = parent_dir + "/results/" + output_dir 
	createFolder(output_path)

	test_data_name = ""
	for ele in sub_folders:
		test_data_name += (ele + ",")

	images_path = []
	masks_path = []
	imgMask_names = []

	for k in range(len(csv_filename)): 
		dfs = []
		for ele in sub_folders: 
			df = pd.read_csv(parent_dir + "/" + test_data_folder + "/" + ele +"/" + csv_filename[k])
			dfs.append(df)

	
	
		for i in range(len(dfs)): 
			for name in zip(dfs[i]["Images"], dfs[i]["Hub Y"], dfs[i]["Hub X"], dfs[i]["Tip Y"], dfs[i]["Tip X"],dfs[i]["Hub Present"]): 
				if name[3]:
					label_ele = [] 
					imgMask_names.append( name[0])
					images_path.append(parent_dir + "/" + test_data_folder + "/" + sub_folders[i] + "/" + img_dir[k] + "/" + name[0])
					label_ele.append(parent_dir + "/" + test_data_folder + "/" + sub_folders[i] + "/" + mask_dir[k] + "/" + name[0])
					label_ele.append(name[1:])
					masks_path.append(label_ele)


	mse_losses = []

	test_dataset = cid.Custom_ImageRegressor_Dataset2(images_path, masks_path, ifgray = False, ifTrain = False, ifFullRange = FullRange, ifAddMskCha = ifAddMskCha)

	model = torch.jit.load(parent_dir + "/models/hub_regression/" + model_name)
	model.to(device)
	model.eval() 
	print("Start")
	n = 0 
	fps = []
	reds = 0 
	blues = 0
	whites = 0
	for data in test_dataset:
		
		image, true_yx, true_p = data

		cha_, height_, width_ = image.size() 
		image_gpu = image.to(device)

		with torch.no_grad(): 
			start_time = time.time() 

			if ifOnlyRegA:
				pred_yx = model(torch.reshape(image_gpu,(1,cha_,height_,width_)))
			else: 
				pred_yx, pred_p = model(torch.reshape(image_gpu,(1,cha_,height_,width_)))

			ifps = 1 / ((time.time() - start_time) + 0.00001)
			fps.append(ifps)
			print("Fequency: ", ifps)

			pred_yx = pred_yx.cpu()
			
			if FullRange == False:  
				true_yx[0] = true_yx[0] * height_
				true_yx[1] = true_yx[1] * width_
				pred_yx[0][0] = pred_yx[0][0] * height_
				pred_yx[0][1] = pred_yx[0][1] * width_
            
# 			print(pred_yx[0])
# 			print(true_yx)
# 			print(pred_yx[0][:2])
			l2 = MSE(pred_yx[0], true_yx) 
			mse_losses.append(l2)

		input_np = np.transpose(image.numpy(), (1,2,0))
		
		y_loc = int(np.round(pred_yx[0][0].numpy()))
		x_loc = int(np.round(pred_yx[0][1].numpy()))
		input_np = input_np * 255
		if (l2 < 5):
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2] = 255
			whites += 1
		elif (l2 < 10): 
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2,0] = 0
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2,1] = 125
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2,2] = 255
			blues += 1
		else: 
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2,0] = 255
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2,1] = 0
			input_np[y_loc - 2: y_loc + 2, x_loc - 2: x_loc + 2,2] = 0
			reds += 1
            
		input_np = cv2.cvtColor(input_np, cv2.COLOR_BGR2RGB)

		cv2.imwrite(output_path + "/" + imgMask_names[n], input_np)
		n += 1 
	
	print("Mean FPS: ", np.mean(fps[2:]))
		
	field_names = ['Date', 'Test Data', 'Output Dir', 'Model Name', 'MSE error', 'MSE std','MSE min', 'MSE max', 'Whites', 'Blues', 'Reds', 'Total']

	t_ = datetime.datetime.now()
	date = str(t_.month) + "/" + str(t_.day) + "/" + str(t_.year)
	mse_error = np.mean(mse_losses)
	mse_std = np.std(mse_losses)
	mse_min = np.min(mse_losses)
	mse_max = np.max(mse_losses)

	total = reds + blues + whites
    
	dict = {'Date':date, 'Test Data':test_data_name, 'Output Dir':output_dir, 'Model Name': model_name, 'MSE error':mse_error, 
				'MSE std':mse_std, 'MSE min':mse_min, 'MSE max':mse_max, 'Whites': whites, 'Blues' : blues, 'Reds' : reds, 'Total' : total}
	result_csv = parent_dir + "/results/regressionModelResult.csv"

	with open(result_csv, 'a') as f_object:
		dictwriter_object = DictWriter(f_object, fieldnames=field_names)
		dictwriter_object.writerow(dict)
		f_object.close()





if __name__ == "__main__": 
	
	print("Test Hub Localization Model")


	model_name = "model_cyc_1_ep_400_bs_128_lr_0.001_20240123_T155510.pt"
	output_dir = "aaaa"
	parent_dir = "//cgmqnap.clearguide.local/data/Needles/Lumena"  

	test_data_folder = "test_data"

	sub_folders = ["CIRS22", "EL", "EL_2", "PC2", "CNMC", "PR"]
	csv_filename = [ "cropped_256x256.csv"]
	img_dir = [ "images_cropped_256x256"]
	mask_dir = [ "masksLined_cropped_256x256"]

	threshold = 0.7
	FullRange = False 
	ifAddMskCha = True 
	ifOnlyRegA = False

	Test_HubRegression(model_name = model_name, output_dir = output_dir, parent_dir = parent_dir,
				test_data_folder = test_data_folder, sub_folders= sub_folders, img_dir = img_dir, 
				mask_dir = mask_dir, csv_filename = csv_filename, FullRange = FullRange, 
                ifAddMskCha = ifAddMskCha,ifOnlyRegA = ifOnlyRegA)