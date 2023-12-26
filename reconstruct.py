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
import utils

from PIL import Image


def reconstruct(device, model_fname, dataset_name, num_latent_dims, rec_testdata):
       
    # Load the training and test data
    batch_size = 32

    # Image size
    img_size = (32, 32)

    # get the data
    train_loader, test_loader, _ , num_img_channels = dataset.get_loaders(dataset_name, img_size, batch_size)

    if rec_testdata:
        data_loader = test_loader
        suffix = "test"
    else:
        data_loader = train_loader
        suffix = "train"

    # Load the model
    vae = model.VAE(num_latent_dims, num_img_channels, device=device)
    vae.load(model_fname)
    print(f"Loaded model with {num_latent_dims} latent dims from {model_fname}")

    # push the model to the device we are using
    vae.to(device)

    # loop over data and reconstruct
    with torch.no_grad():
        img_count = 0
        img_path = f"./reconstructions/{dataset_name}_{suffix}/{num_latent_dims:04d}_latent_dims/img_"
        utils.ensure_folder_exists(img_path)
        
        for i, data in enumerate(data_loader, 0):        
                # get the testing data and push the data to the device we are using       
                images = data[0].to(device)

                # reconstruct the images
                images_recon = vae(images)

                # save the images
                for j in range(images.shape[0]):
                    img1 = images[j]
                    img2 = images_recon[j]

                     # Convert PyTorch tensors to numpy arrays and scale to 0-255
                    img1_data = (img1.detach().cpu().numpy() * 255).astype(np.uint8)
                    img2_data = (img2.detach().cpu().numpy() * 255).astype(np.uint8)

                     # filenames for image and reconstructed image
                    img_fname = f"{img_path}{(img_count+j):08d}.png"
                    utils.combine_and_save_image(img1_data, img2_data, img_fname)
                 
                
                # print progress
                print(f"Reconstructed {img_count+images.shape[0]} images", end="\r", flush=True)
                
                # update image count
                img_count = img_count + images.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a VAE with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")     
    parser.add_argument('--model', type=str, required=True, help='Model filename *.pth')
    parser.add_argument('--rec_testdata', action="store_true", help="Reconstruct test split instead of training split")  
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar-10', 'cifar-100', 'celeb-a'], default='mnist', 
                        help="Select the dataset to use (mnist, cifar-10, cifar-100, celeb-a)") 
    parser.add_argument("--latent_dims", type=int, required=True, help="Number of latent dimensions (positive integer)")


    args = parser.parse_args()

    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device('cpu')
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    if not args.rec_testdata:
        print("Reconstructing training data")
    else:
        print("Reconstructing test data")

    reconstruct(dev, args.model, args.dataset, args.latent_dims, args.rec_testdata)
