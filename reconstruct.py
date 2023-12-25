# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import dataset
import model
import device

from PIL import Image

def save_image(img, fname):
    img_data = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img_data = img_data.squeeze()

    # Save the image using Pillow
    pimg = Image.fromarray(img_data)
    pimg.save(fname)


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
        img_path = "./orig_img/img_"
        recon_img_path = "./recon_img/img_recon_"
        for i, data in enumerate(test_loader, 0):        
                # get the testing data and push the data to the device we are using       
                images = data[0].to(device)

                # reconstruct the images
                images_recon = vae(images)

                # save the images
                for j in range(images.shape[0]):
                     # filenames for image and reconstructed image

                    img_fname       = f"{img_path}      {(img_count+j):08d}.png"
                    recon_img_fname = f"{recon_img_path}{(img_count+j):08d}.png"
                   
                    save_image(images[j],       img_fname)
                    save_image(images_recon[j], recon_img_fname)
                
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
