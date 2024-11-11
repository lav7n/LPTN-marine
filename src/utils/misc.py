import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import nibabel as nib
from PIL import Image

# normalizing target image to be compatible with tanh activation function
def normalize_data(data):
    data *= 2
    data -= 1
    return data

def unnormalize_data(data):
    data += 1
    data /= 2
    return data

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def list_img(directory):
    image_extensions = ['.png', '.jpg']
    image_paths = []

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            full_path = os.path.join(directory, filename)
            image_paths.append(full_path)

    return sorted(image_paths)

