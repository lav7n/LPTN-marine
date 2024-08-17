import numpy as np
import cv2
from .misc import list_img
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

        # mean and std computed for normalized input
        self.mean = np.array([0.63799969, 0.67506404, 0.59012203], dtype=np.float32)
        self.std = np.array([0.21021621, 0.20920322, 0.20019643], dtype=np.float32)

    def __getitem__(self, i):
        image = cv2.imread(self.images_list[i])
        image = image.reshape(384, 512, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_array = cv2.imread(self.masks_list[i], 0)
        mask = mask_array.reshape(384, 512, 1)

        # Map all values of 4 to 3 in the mask
        mask_array[mask_array == 4] = 3
        
        # Reshape mask after mapping
        mask = mask_array.reshape(384, 512, 1)

        # Print unique values after mapping
        unique_values_after = np.unique(mask_array)
        # print(f"Unique values in mask {i} after mapping: {unique_values_after}")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
        if self.preprocessing:
            image = image / 255.0
            image = (image - self.mean) / self.std
        
        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')

        return image, mask 
        
    def __len__(self):
        return len(self.images_list)

