"""

Custom Dataset class for dataloader

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""
import torch
import random 
from torch.utils.data import Dataset 
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ColorJitter 
from torchvision.transforms.functional import hflip, vflip, rotate 
from torchvision.transforms.v2 import GaussianBlur
              
    
class NeedleDataset(Dataset): 
    def __init__(self, images_path, labels, ifSegmentationTrain = False, ifgray = False, ifTrain = True, ifFullRange = False, ifAddMskCha = False): 
        
        self.images_path = images_path
        self.labels = labels
        self.ifSegmentationTrain = ifSegmentationTrain
        self.ifgray = ifgray
        self.ifTrain = ifTrain 
        self.ifFullRange = ifFullRange
        self.ifAddMskCha = ifAddMskCha
        
        if self.ifSegmentationTrain:
            self.ifAddMskCha = False
        
    def __len__(self): 
        return len(self.images_path)
    
    def transform(self, image, mask, h_y, h_x, t_y, t_x): 
        
        _, h, w = image.size() 
        
        if random.random() < 0.5: 
            image = hflip(image)
            mask = hflip(mask)  
            
            if not self.ifSegmentationTrain: 
                if self.ifFullRange: 
                    h_x = w - 1 - h_x
                    t_x = w - 1 - t_x
                else:     
                    h_x = 1 - (1/w) - h_x
                    t_x = 1 - (1/w) - t_x
                
     
        jitter = ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
        image = jitter(image)  
        
        if random.random() > 0.10: 
            blur = GaussianBlur(9)
            image = blur(image)
           
        return image, mask, h_y, h_x, t_y, t_x

    def __getitem__(self, idx):
        
        if self.ifgray: 
            img = read_image(self.images_path[idx], mode=ImageReadMode.GRAY)
            img = img.type(torch.FloatTensor)
            img = img / 255
        else: 
            img = read_image(self.images_path[idx],mode=ImageReadMode.RGB)
            img = img.type(torch.FloatTensor)
            img = img / 255
               
        
        if self.ifSegmentationTrain or self.ifAddMskCha: 
            msk = read_image(self.labels[idx][0])
            msk = msk.type(torch.FloatTensor)
        else: 
            msk = None
            
        
        _, h, w = img.size() 
        
        if not self.ifSegmentationTrain: 
            if self.ifFullRange: 
                h_y = self.labels[idx][1][0]
                h_x = self.labels[idx][1][1]
                t_y = self.labels[idx][1][2]
                t_x = self.labels[idx][1][3]
            else: 
                h_y = self.labels[idx][1][0]/h 
                h_x = self.labels[idx][1][1]/w
                t_y = self.labels[idx][1][2]/h
                t_x = self.labels[idx][1][3]/w
                
            torch.tensor(h_y)
            torch.tensor(h_x)
            hub_present = torch.tensor(hub_present)
            hub_present = torch.reshape(hub_present, (1,))
                
            hub_present = self.labels[idx][1][4]
        else: 
            h_y = None 
            h_x = None 
            t_x = None 
            t_y = None
        
        if self.ifTrain: 
            img, msk, h_y, h_x, t_y, t_x = self.transform(img, msk, h_y, h_x, t_y, t_x)
          
        if self.ifAddMskCha:
            img = torch.cat((img,msk))
        
        
        if self.ifTrain: 
            if self.ifSegmentationTrain:
                return img, msk
            else:
                
                return img, torch.tensor((h_y, h_x, t_y, t_x)), hub_present
        else: 
            return img 

    
    
    
class CustomImageGrayMaskDataset_dataAugmentation(Dataset): 
    def __init__(self, images_path , masks_path, ifgray = False, ifTrain = False): 
        self.images_path = images_path
        self.masks_path = masks_path
        self.ifgray = ifgray
        self.ifTrain = ifTrain
        
    def __len__(self): 
        return len(self.images_path)
    
    def transform(self, image, mask): 
        
        if random.random() > 0.50: 
            image = hflip(image)
            mask = hflip(mask)


        if random.random() > 0.50:
            deg = random.randint(-45,45)
            image = rotate(image, deg)
            mask = rotate(mask, deg)
            
        
        jitter = ColorJitter(brightness = 0.2)
        image = jitter(image)
        
        if random.random() > 0.10: 
            blur = GaussianBlur(9)
            image = blur(image)

        return image, mask 

    def __getitem__(self, idx):
        
        if self.ifgray: 
            img = read_image(self.images_path[idx], mode=ImageReadMode.GRAY)
            img = img / 255
        else: 
            img = read_image(self.images_path[idx],mode=ImageReadMode.RGB)
            img = img.type(torch.FloatTensor)
            img = img / 255
        
        if self.masks_path: 
            msk = read_image(self.masks_path[idx])
            msk = msk.type(torch.FloatTensor)
        
        if self.ifTrain: 
            img, msk = self.transform(img, msk)
        
        if self.masks_path: 
            return img, msk  
        else: 
            return img
        
class CustomImageGray_2MaskDataset_dataAugmentation(Dataset): 
    def __init__(self, images_path , masks_path, ifgray = False, ifTrain = False): 
        self.images_path = images_path
        self.masks_path = masks_path
        self.ifgray = ifgray
        self.ifTrain = ifTrain
        
    def __len__(self): 
        return len(self.images_path)
    
    def transform(self, image, mask1, mask2): 
        
        if random.random() > 0.50: 
            image = hflip(image)
            mask1 = hflip(mask1)
            mask2 = hflip(mask2)
            


        if random.random() > 0.50:
            deg = random.randint(-45,45)
            image = rotate(image, deg)
            mask1 = rotate(mask1, deg)
            mask2 = rotate(mask2, deg)
            
        
        jitter = ColorJitter(brightness = 0.2)
        image = jitter(image)
        
        if random.random() > 0.30: 
            blur = GaussianBlur(9)
            image = blur(image)

        return image, mask1, mask2

    def __getitem__(self, idx):
        
        if self.ifgray: 
            img = read_image(self.images_path[idx], mode=ImageReadMode.GRAY)
            img = img / 255
        else: 
            img = read_image(self.images_path[idx],mode=ImageReadMode.RGB)
            img = img.type(torch.FloatTensor)
            img = img / 255
        
        if self.masks_path: 
            msk1 = read_image(self.masks_path[idx][0])
            msk1 = msk1.type(torch.FloatTensor)
            
            msk2 = read_image(self.masks_path[idx][1])
            msk2 = msk2.type(torch.FloatTensor)
            

            
        
        if self.ifTrain: 
            img, msk1, msk2 = self.transform(img, msk1, msk2)
        
        if self.masks_path: 
            return img, msk1, msk2
        else: 
            return img

    
    
class Custom_ImageRegressor_Dataset(Dataset): 
    def __init__(self, images_path , labels, ifgray = False, ifTrain = True, ifFullRange = False, ifAddMskCha = False): 
        self.images_path = images_path
        self.labels = labels
        self.ifgray = ifgray
        self.ifTrain = ifTrain 
        self.ifFullRange = ifFullRange
        self.ifAddMskCha = ifAddMskCha
        
    def __len__(self): 
        return len(self.images_path)
    
    def transform(self, image, mask, h_y, h_x): 
        
        _, h, w = image.size() 
        
        if random.random() < 0.5: 
            image = hflip(image)
            mask = hflip(mask)  
            
            if self.ifFullRange: 
                h_x = w - 1 - h_x
            else:     
                h_x = 1 - (1/w) - h_x
                
     
        jitter = ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
        image = jitter(image)  
        
        if random.random() > 0.10: 
            blur = GaussianBlur(9)
            image = blur(image)
#             mask = blur(mask)

        return image, mask, h_y, h_x

    def __getitem__(self, idx):
        
        if self.ifgray: 
            img = read_image(self.images_path[idx], mode=ImageReadMode.GRAY)
            img = img / 255
        else: 
            img = read_image(self.images_path[idx],mode=ImageReadMode.RGB)
            img = img.type(torch.FloatTensor)
            img = img / 255
               
         
        msk = read_image(self.labels[idx][0])
        msk = msk.type(torch.FloatTensor)
            
        
        _, h, w = img.size() 
        
        
        if self.ifFullRange: 
            h_y = self.labels[idx][1][0]
            h_x = self.labels[idx][1][1] 
        else: 
            h_y = self.labels[idx][1][0]/h 
            h_x = self.labels[idx][1][1]/w
            
        hub_present = self.labels[idx][1][2]
        
        if self.ifTrain: 
            img, msk, h_y, h_x = self.transform(img, msk, h_y, h_x)
          
        if self.ifAddMskCha:
            img = torch.cat((img,msk))

        torch.tensor(h_y)
        torch.tensor(h_x)
        hub_present = torch.tensor(hub_present)
        hub_present = torch.reshape(hub_present, (1,))
               
        return img, torch.tensor((h_y, h_x)), hub_present
    
    
class Custom_ImageRegressor_Dataset2(Dataset): 
    def __init__(self, images_path , labels, ifgray = False, ifTrain = True, ifFullRange = False, ifAddMskCha = False): 
        self.images_path = images_path
        self.labels = labels
        self.ifgray = ifgray
        self.ifTrain = ifTrain 
        self.ifFullRange = ifFullRange
        self.ifAddMskCha = ifAddMskCha
        
    def __len__(self): 
        return len(self.images_path)
    
    def transform(self, image, mask, h_y, h_x, t_y, t_x): 
        
        _, h, w = image.size() 
        
        if random.random() < 0.5: 
            image = hflip(image)
            mask = hflip(mask)  
            
            if self.ifFullRange: 
                h_x = w - 1 - h_x
                t_x = w - 1 - t_x
            else:     
                h_x = 1 - (1/w) - h_x
                t_x = 1 - (1/w) - t_x
                
     
        jitter = ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
        image = jitter(image)  
        
        if random.random() > 0.10: 
            blur = GaussianBlur(9)
            image = blur(image)
#             mask = blur(mask)

        return image, mask, h_y, h_x, t_y, t_x

    def __getitem__(self, idx):
        
        if self.ifgray: 
            img = read_image(self.images_path[idx], mode=ImageReadMode.GRAY)
            img = img / 255
        else: 
            img = read_image(self.images_path[idx],mode=ImageReadMode.RGB)
            img = img.type(torch.FloatTensor)
            img = img / 255
               
         
        msk = read_image(self.labels[idx][0])
        msk = msk.type(torch.FloatTensor)
            
        
        _, h, w = img.size() 
        
        
        if self.ifFullRange: 
            h_y = self.labels[idx][1][0]
            h_x = self.labels[idx][1][1]
            t_y = self.labels[idx][1][2]
            t_x = self.labels[idx][1][3]
        else: 
            h_y = self.labels[idx][1][0]/h 
            h_x = self.labels[idx][1][1]/w
            t_y = self.labels[idx][1][2]/h
            t_x = self.labels[idx][1][3]/w
            
        hub_present = self.labels[idx][1][4]
        
        if self.ifTrain: 
            img, msk, h_y, h_x, t_y, t_x = self.transform(img, msk, h_y, h_x, t_y, t_x)
          
        if self.ifAddMskCha:
            img = torch.cat((img,msk))

        torch.tensor(h_y)
        torch.tensor(h_x)
        hub_present = torch.tensor(hub_present)
        hub_present = torch.reshape(hub_present, (1,))
               
        # return img, torch.tensor((h_y, h_x)), torch.tensor((t_y, t_x)), hub_present
        return img, torch.tensor((h_y, h_x, t_y, t_x)), hub_present
    
    
    
    
class HubLocalizationOneHotDataset(Dataset): 
    def __init__(self, images_path , labels, ifgray = False, ifTrain = True, ifAddMskCha = False): 
        self.images_path = images_path
        self.labels = labels
        self.ifgray = ifgray
        self.ifTrain = ifTrain 
        self.ifAddMskCha = ifAddMskCha
        
    def __len__(self): 
        return len(self.images_path)
    
    def transform(self, image, mask, h_y, h_x): 
        
        _, h, w = image.size() 
        
        if random.random() < 0.5: 
            image = hflip(image)
            mask = hflip(mask)  
            h_x = w - 1 - h_x
     
        jitter = ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
        image = jitter(image)  
        
        if random.random() > 0.10: 
            blur = GaussianBlur(9)
            image = blur(image)
            mask = blur(mask)

        return image, mask, h_y, h_x

    def __getitem__(self, idx):
        
        if self.ifgray: 
            img = read_image(self.images_path[idx], mode=ImageReadMode.GRAY)
            img = img / 255
        else: 
            img = read_image(self.images_path[idx],mode=ImageReadMode.RGB)
            img = img.type(torch.FloatTensor)
            img = img / 255
               
         
        msk = read_image(self.labels[idx][0])
        msk = msk.type(torch.FloatTensor)
           
        _, height, widht = img.size()    

        h_y = self.labels[idx][1][0]
        h_x = self.labels[idx][1][1] 
        hub_present = self.labels[idx][1][2]
        
        if self.ifTrain: 
            img, msk, h_y, h_x = self.transform(img, msk, h_y, h_x)
          
        
        if self.ifAddMskCha:
            img = torch.cat((img,msk))

        
        y_oneHot = torch.zeros(height)
        y_oneHot[h_y] = 1.0
        
        x_oneHot = torch.zeros(width)
        x_oneHot[h_x] = 1.0
        
        hub_present = torch.tensor(hub_present)
        hub_present = torch.reshape(hub_present, (1,))
               
            
        return img, y_oneHot, x_oneHot, hub_present
    

    
    
    
    
    