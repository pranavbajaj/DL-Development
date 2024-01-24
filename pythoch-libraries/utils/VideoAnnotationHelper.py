"""

Helper funtion for Vidoe Annotation 

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""

import os 
import cv2
import glob 
import numpy as np
from sklearn.linear_model import HuberRegressor, RANSACRegressor, LinearRegression 



def createFolder(path="D:/lol"):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder '{path}'.")
    else:
        print(f"Folder '{path}' already exists.")
        

def ImportImages(path = "D:\\",
                iformat = "png"):
    """
    Imports images of given format from the path location.
    
    Attributes: 
    path: Location of the folder where images are stored. 
    iformat: Input image format.
    
    return: Numpy array of images 
    """
    
    if not os.path.exists(path): 
        raise Exception("Path to input images is invalid!")
        
    image_paths = path + "\\*." + iformat
    images = [] 
    
    # Check if there exist images at the path of sepcified format  
    if np.shape(glob.glob(image_paths))[0] == 0:
        raise Exception("No " + iformat + " images found at the input path")   
    
    for filename in glob.glob(image_paths): 
        img = cv2.imread(filename)
        images.append(img)
            
    return images


def GrayImages(images,
              img_type = "bgr"): 
    
    """
    Converts array of RGB or BGR images to gray images
    
    Attributes: 
    images: Numpy array of images. 
    img_type: Image channel format. Ex: bgr or rgb 
    
    return: Numpy array of gray images 
    """
    
    if images[0].ndim != 3:
        raise Exception("Images passed to GrayImages() is not 3-dimentional")
    
    if img_type == "bgr":
        convertor = cv2.COLOR_BGR2GRAY
    elif img_type == "rgb": 
        convertor = cv2.COLOR_RGB2GRAY
    else: 
        raise Exception("Unknown image format")         
        
    gray_images = [] 
    
    for img in images:
        img_gray = cv2.cvtColor(img, convertor)
        gray_images.append(img_gray)    
        
    return gray_images 


def LineFitOnMask(masked_image, threshold = 0.5): 
    
    """
    Does HuberRegressor line fit on Masked image. 
    
    Attributes: 
    masked_image: Single gray image. 
    
    return: Trained regressor. 
    """
    
    if masked_image.ndim != 2: 
        raise Exception("Image passed to LineFitMask() is not a gray image")
    
    y, X = np.where(masked_image>threshold)
    X = np.expand_dims(X, axis=1)
    
    regressor = None 
    if np.shape(y)[0] != 0:
        regressor = RANSACRegressor().fit(X,y)
    
        
    return regressor


def LineMaskUint(regressor, 
                 height = 853,
                 width = 480,
                 color = "r"): 
    
    """
    Return a annotated image based on regressor 
    
    Attribute: 
    regressor: regressor trained on the Masked gray image.
    height: Height of required annotated image. 
    widht: Width of required annotated image. 
    color: Required annotation color channel 
    
    return: Annotated color image.
    
    """
    
    if height < 0 or width < 0: 
        raise Exception("Height or Width are not > 0")
    
    if type(height) != int or type(width) != int: 
        raise Exception("Height or width passed are not int")
    
    masked_output = np.zeros([height, width], dtype=np.uint8)
    c = np.zeros([height, width], dtype=np.uint8)
    
    for i in range(width):
        y_pred = regressor.predict([[i]])
        if int(y_pred) >= 0 and int(y_pred) < height: 
            masked_output[int(y_pred)][i] = 255
    
    # Note: cv2.merge takes r,g,b order 
    if color == "r": 
        mask = cv2.merge((masked_output, c, c))
    elif color == "g": 
        mask = cv2.merge((c, masked_output, c))
    elif color == "b":
        mask = cv2.merge((c, c, masked_output))
    else: 
        raise Exception("Invalid color channel for annotated image")
        
    
    return mask


def LineMaskfloat32(regressor, 
                 height = 853,
                 width = 480,
                 color = "r"): 
    
    """
    Return a annotated image based on regressor 
    
    Attribute: 
    regressor: regressor trained on the Masked gray image.
    height: Height of required annotated image. 
    widht: Width of required annotated image. 
    color: Required annotation color channel 
    
    return: Annotated color image.
    
    """
    
    if height < 0 or width < 0: 
        raise Exception("Height or Width are not > 0")
    
    if type(height) != int or type(width) != int: 
        raise Exception("Height or width passed are not int")
    
    masked_output = np.zeros([height, width], dtype=np.float32)
    c = np.zeros([height, width], dtype=np.float32)
    
    for i in range(width):
        y_pred = regressor.predict([[i]])
        if round(y_pred[0]) >= 0 and round(y_pred[0]) < height: 
            masked_output[round(y_pred[0])][i] = 1.0

    m = regressor.coef_[0]
    c = regressor.intercept_

    for y_pred in range(height):
        x_pred = round((y_pred - c)/m)
        if x_pred >=0 and x_pred < width: 
            masked_output[y_pred][x_pred] = 1.0 
    
    # Note: cv2.merge takes r,g,b order 
    # if color == "r": 
    #     mask = cv2.merge((masked_output, c, c))
    # elif color == "g": 
    #     mask = cv2.merge((c, masked_output, c))
    # elif color == "b":
    #     mask = cv2.merge((c, c, masked_output))
    # else: 
    #     raise Exception("Invalid color channel for annotated image")
        
    
    return masked_output

    
def SaveImage(img, 
             output_loc = "D:\\",
             ofile_name = "output",
             oformat = "png"):
    
    """
    Saves input at the specified location 
    
    Attributes: 
    img: image (numpy array). 
    output_loc: Location at which the image should be saved.
    ofile_name: Name of the saved image. 
    oformat: Format of saved image. Ex: png, jpg, etc.
    
    return: None 
    """
    
    if not os.path.exists(output_loc):
        print("Output folder location not found, creating the folder")
        os.makedirs(output_loc)
    
    output_path = output_loc + "\\" + ofile_name + "." + oformat
    cv2.imwrite(output_path, img)