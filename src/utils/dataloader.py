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
        self.mean = np.array([0.24173183, 0.43415226, 0.49751509], dtype=np.float32)
        self.std = np.array([0.17116954, 0.18768028, 0.20497839], dtype=np.float32)

        self.scaled_rgb_to_class = {
            (0, 0, 0): 0,         # Background - Black
            (255, 0, 124): 1,     # Oil - Custom Pink
            (255, 204, 51): 3,    # Others - Yellow-Orange
            (51, 221, 255): 2     # Water - Light Blue
        }

    def __getitem__(self, i):
        image = cv2.imread(self.images_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512,512))  # Resize the image to (width, height)

        mask = cv2.imread(self.masks_list[i])
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, (512,512))  # Resize the mask_rgb to (width, height)

        mask_mapped = np.zeros((512,512), dtype=np.uint8)

        for rgb, cls in self.scaled_rgb_to_class.items():
            mask_mapped[(mask_rgb == rgb).all(axis=2)] = cls

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_mapped)
            image, mask_mapped = sample['image'], sample['mask']
 
        image = image / 255.0
        # image = (image - self.mean) / self.std
    
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # Convert to (C, H, W) format
        mask_mapped = torch.from_numpy(mask_mapped).unsqueeze(0).long()  # Add channel dimension

        return image, mask_mapped
        
    def __len__(self):
        return len(self.images_list)

