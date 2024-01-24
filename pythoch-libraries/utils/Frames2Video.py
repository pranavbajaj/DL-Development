"""
Combines images/frames into a video 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import cv2
import numpy as np 
import glob 
import os 

def f2v(path = "D:\\data\\lumena\\one\\R",
        output_loc="D:\\needle_detection\\XMem", 
        ofile_name = "output",
        iformat = "png",
        oformat = "mp4",
        ofps = 60):
    
    """
    Converts frames to a video. 
    
    Attributes: 
    path = path to a folder which contain input images.
    output_loc = location where output video should be stored. 
    ofile_name = output video file name.
    iformat = input image format. Ex: jpg, png, etc. 
    oformat = output video format. Ex: mp4.
    ofps = output video's fps. 
    
    return: Video of specified format.
    """
    
    # Check if the path specified to the input images is valid 
    if not os.path.exists(path):
         raise Exception("Path to input images is invalid!")
    
    frame_paths = path + "\\*." + iformat 
    frame_array = []
   
    # Check if there exist images at the path of sepcified format  
    if np.shape(glob.glob(frame_paths))[0] == 0:
        raise Exception("No " + iformat + " images found at the input path")
       
    for filename in glob.glob(frame_paths): 
        img = cv2.imread(filename)
        frame_array.append(img)
        
    height, width, layers = img.shape
    size = (width, height)
    
    
    if oformat == "mp4": 
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    else: 
        raise Exception("Unknow output file formate")
        
    if not os.path.exists(output_loc):
        print("Output folder location not found, creating the folder")
        os.makedirs(output_loc) 
        
    output_path = output_loc + "\\" + ofile_name + "." + oformat
    out = cv2.VideoWriter(output_path, fourcc, ofps, size)
    
    for i in range(len(frame_array)):
        out.write(frame_array[i])
        
    print("Video created: ", output_path)
    
    out.release() 
    
    
    
    
    