import numpy as np
import cv2
from .misc import list_img
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
from sklearn.model_selection import train_test_split
import os
import torch
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=True):
        self.images_list = images_dir
        self.masks_list = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # Define the scaled RGB to class index mapping
        self.scaled_rgb_to_class = {
            (0, 0, 0): 0,       # Background (waterbody) - Black
            (0, 0, 255): 1,     # Human divers - Blue
            (0, 255, 0): 2,     # Aquatic plants and sea-grass - Green
            (0, 255, 255): 3,   # Wrecks and ruins - Sky
            (255, 0, 0): 4,     # Robots (AUVs/ROVs/instruments) - Red
            (255, 0, 255): 5,   # Reefs and invertebrates - Pink
            (255, 255, 0): 6,   # Fish and vertebrates - Yellow
            (255, 255, 255): 7  # Sea-floor and rocks - White
        }

    def __getitem__(self, i):
        image = cv2.imread(self.images_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_list[i])
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask_mapped = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

        for rgb, cls in self.scaled_rgb_to_class.items():
            mask_mapped[(mask_rgb == rgb).all(axis=2)] = cls

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_mapped)
            image, mask_mapped = sample['image'], sample['mask']
    
        image = image / 255.0
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_mapped = torch.from_numpy(mask_mapped).unsqueeze(0).long()

        return image, mask_mapped
        
    def __len__(self):
        return len(self.images_list)

