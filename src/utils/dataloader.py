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
# to_tensor = transforms.ToTensor()
      

class Dataset(BaseDataset):
    def __init__(
            self, 
            images_dir: list, 
            masks_dir: list, 
            augmentation=None, 
            preprocessing=None,
    ):

        self.images_list = images_dir
        self.masks_list = masks_dir
        # self.classes = {'Background':0, 'TUM':1, 'STR':2, 'LYM':3, 'NEC':4}

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    # def standardize(self,image,mean,std):
    #     image = image/255.0
    #     image_normalised = image - mean
    #     image_normalised = image_normalised / std
    #     return image_normalised

    def __getitem__(self, i):

        img_array = cv2.imread(self.masks_list[i])
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        color_to_class = {
            (255, 0, 0): 1,  # TUM
            (0, 255, 0): 2,  # STR
            (0, 0, 255): 3,  # LYM
            (153, 0, 255): 4,  # NEC
            (255, 255, 255): 0  # Background or exclude
        }

        single_channel_array = np.zeros((224, 224), dtype=np.uint8)

        for color, class_value in color_to_class.items():
            mask = np.all(img_array == np.array(color, dtype=np.uint8), axis=-1)
            single_channel_array[mask] = class_value

        mask = single_channel_array
        mask = mask.reshape(224,224,1)
        # new_mask = np.zeros((224, 224, 5), dtype=np.uint8)
        # for class_value in range(5):
        #      new_mask[..., class_value] = (mask[..., 0] == class_value)
        # mask = new_mask

        # # mask = mask.astype(np.int64)
        image = cv2.imread(self.images_list[i])
        image = image.reshape(224,224,3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype(np.int64)
        #Right now mask is (224,224,1) array, image is (224,224,3) array
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            #sample = self.preprocessing(image=image, mask=mask)
            #image, mask = sample['image'], sample['mask']
            image = image.transpose(2, 0, 1).astype('float32')
            mask = mask.transpose(2, 0, 1).astype('float32')
            # image[0] = self.standardize(image[0],0.519624,0.24610637)
            # image[1] = self.standardize(image[1],0.34783007,0.25106501)
            # image[2] = self.standardize(image[2],0.68143765,0.18394224)
            image = image/255.0

            
        mask = mask.reshape(224,224)
        return image, mask #label
        
    def __len__(self):
        return len(self.images_list)
