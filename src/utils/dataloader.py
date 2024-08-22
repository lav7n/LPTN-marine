import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from sklearn.model_selection import train_test_split
import os
import torch
from torchvision import transforms
from PIL import Image

class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=True):
        self.images_list = images_dir
        self.masks_list = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, i):
        image = cv2.imread(self.images_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320,256))  

        mask = cv2.imread(self.masks_list[i], 0)
        mask = cv2.resize(mask, (320,256)) 

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
        image = image / 255.0
        mask = mask/255.0 # Mask values are 0,255

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # Convert to (C, H, W) format
        mask = torch.from_numpy(mask).unsqueeze(0).long() 

        return image, mask
        
    def __len__(self):
        return len(self.images_list)




