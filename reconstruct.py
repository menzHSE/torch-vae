# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import dataset
import model
import device

from PIL import Image

def ensure_folder_exists(img_path):
    # Extract the directory path from the file path
    dir_path = os.path.dirname(img_path)

    # Check if the directory path is not empty
    if dir_path:
        # Check if the directory exists, and create it if it does not
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    return dir_path  # Optionally return the directory path

def save_image(img1, img2, fname):
    # Convert PyTorch tensors to numpy arrays and scale to 0-255
    img1_data = (img1.detach().cpu().numpy() * 255).astype(np.uint8)
    img2_data = (img2.detach().cpu().numpy() * 255).astype(np.uint8)

    # Remove any extra dimensions (like channels in grayscale images)
    img1_data = img1_data.squeeze()
    img2_data = img2_data.squeeze()

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


def reconstruct(device, model_fname, dataset_name, num_latent_dims):
       
    # Load the training and test data
    batch_size = 32
    # get the data
    _, test_loader, _ = dataset.mnist(batch_size)


    # Load the model
    vae = model.VAE(num_latent_dims, device=device)
    vae.load(model_fname)
    print(f"Loaded model with {num_latent_dims} latent dims from {model_fname}")

    # push the model to the device we are using
    vae.to(device)

    # loop over data and reconstruct
    with torch.no_grad():
        img_count = 0
        img_path = f"./reconstructions/reconstruct_{num_latent_dims:05d}_latent_dims/img_"
        ensure_folder_exists(img_path)
        
        for i, data in enumerate(test_loader, 0):        
                # get the testing data and push the data to the device we are using       
                images = data[0].to(device)

                # reconstruct the images
                images_recon = vae(images)

                # save the images
                for j in range(images.shape[0]):
                     # filenames for image and reconstructed image

                    img_fname = f"{img_path}{(img_count+j):08d}.png"
                    save_image(images[j],  images_recon[j], img_fname)

                
                # print progress
                print(f"Reconstructed {img_count+images.shape[0]} images", end="\r", flush=True)
                
                # update image count
                img_count = img_count + images.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a VAE with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")     
    parser.add_argument('--model', type=str, required=True, help='Model filename *.pth')
    parser.add_argument("--dataset", type=str, choices=['mnist'], default='mnist', 
                        help="Select the dataset to use (mnist)")
    parser.add_argument("--latent_dims", type=int, required=True, help="Number of latent dimensions (positive integer)")


    args = parser.parse_args()

    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device('cpu')
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    reconstruct(dev, args.model, args.dataset, args.latent_dims)
