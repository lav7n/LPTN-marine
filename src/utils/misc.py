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

    # Define paths for images and masks subdirectories
    images_dir = os.path.join(directory, 'images')
    masks_dir = os.path.join(directory, 'masks')

    # Initialize lists for images and masks
    image_paths = []
    mask_paths = []

    # List images from the 'images' subdirectory
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            full_path = os.path.join(images_dir, filename)
            image_paths.append(full_path)

    # List masks from the 'masks' subdirectory
    for filename in os.listdir(masks_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            full_path = os.path.join(masks_dir, filename)
            mask_paths.append(full_path)

    # Sort and return the two lists
    return sorted(image_paths), sorted(mask_paths)


def convert_nifti_to_png(nifti_path, output_dir):
    # Load the NIfTI file
    img = nib.load(nifti_path)
    data = np.array(img.dataobj)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over the slices and save them as PNG files
    for i in range(data.shape[2]):
        # Extract the slice
        slice_data = data[:, :, i]
        
        # Normalize the slice data (optional)
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        
        # Convert the slice to a PIL image
        image = Image.fromarray((slice_data * 255).astype(np.uint8))
        
        # Save the image as PNG
        output_path = os.path.join(output_dir, f"slice_{i}.png")
        image.save(output_path)
        
        print(f"Saved slice {i} as PNG: {output_path}")
        