import os
import numpy as np
from PIL import Image

def save_image(img_data, fname):
    
    # Reshape if the image has a channel dimension
    if img_data.ndim == 3 and img_data.shape[0] == 1:  # Grayscale image
        img_data = img_data.squeeze(0)  # remove channel dimension
    elif img_data.ndim == 3 and img_data.shape[0] == 3:  # Color image
        img_data = img_data.transpose(1, 2, 0)  # C, H, W to H, W, C

    # Convert numpy array to Pillow image
    pimg = Image.fromarray(img_data)

    # Save the image
    pimg.save(fname)


def combine_and_save_image(img1_data, img2_data, fname):
   
    # Reshape if the images have a channel dimension
    if img1_data.ndim == 3 and img1_data.shape[0] == 1:  # Grayscale image
        img1_data = img1_data.squeeze(0)  # remove channel dimension
    elif img1_data.ndim == 3 and img1_data.shape[0] == 3:  # Color image
        img1_data = img1_data.transpose(1, 2, 0)  # C, H, W to H, W, C

    if img2_data.ndim == 3 and img2_data.shape[0] == 1:  # Grayscale image
        img2_data = img2_data.squeeze(0)  # remove channel dimension
    elif img2_data.ndim == 3 and img2_data.shape[0] == 3:  # Color image
        img2_data = img2_data.transpose(1, 2, 0)  # C, H, W to H, W, C

    # Convert numpy arrays to Pillow images
    pimg1 = Image.fromarray(img1_data)
    pimg2 = Image.fromarray(img2_data)

    # Concatenate images horizontally
    total_width = pimg1.width + pimg2.width
    max_height = max(pimg1.height, pimg2.height)
    combined_img = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    combined_img.paste(pimg1, (0, 0))
    combined_img.paste(pimg2, (pimg1.width, 0))

    # Save the combined image
    combined_img.save(fname)
    

def ensure_folder_exists(path):
    # Extract the directory path from the (file) path
    dir_path = os.path.dirname(path)

    # Check if the directory path is not empty
    if dir_path:
        # Check if the directory exists, and create it if it does not
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    return dir_path  # Optionally return the directory path



