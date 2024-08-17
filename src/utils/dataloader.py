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
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.images_list = images_dir
        self.masks_list = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_list[i])
        image = image.reshape(384, 512, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #TODO standardize the input

        mask_array = cv2.imread(self.masks_list[i], 0)
        mask_array = mask_array.reshape(384, 512)

        # Map 4 to 3 in the mask
        mask_array[mask_array == 4] = 3
        
        mask = mask_array.reshape(384, 512, 1)

        print("MASK SHAPE: ", mask.shape)
        print("MASK VALUES", mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')

        image = image / 255.0

        return image, mask 
        
    def __len__(self):
        return len(self.images_list)
