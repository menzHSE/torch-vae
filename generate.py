# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import argparse
import numpy as np
import torch
import torchvision

import model
import device
import utils

def generate(device, model_fname, num_latent_dims, num_img_channels, max_num_filters, num_samples, outdir):

    # Image size
    img_size = (64, 64)

    # Load the model
    vae = model.VAE(num_latent_dims, num_img_channels, max_num_filters, device=device)
    vae.load(model_fname)
    print(f"Loaded model with {num_latent_dims} latent dims from {model_fname}")

    # push the model to the device we are using
    vae.to(device)

    # set model to eval mode
    vae.eval()

    # generate samples
    with torch.no_grad():

        for i in range(num_samples):

            img_path = f"{outdir}/img_{i:06d}.png"
            
            # generate a random latent vector
            
            # during training we have made sure that the distribution in latent
            # space remains close to a normal distribution

            z = torch.randn(num_latent_dims).to(device)

            # generate an image from the latent vector
            img = vae.decode(z)

            if i == 0:
                pics = img
            else:
                pics = torch.cat((pics, img), dim=0)

            # get img data
            img_data = (img.detach().cpu().numpy() * 255).astype(np.uint8).squeeze(0)

            # save the image
            utils.save_image(img_data, img_path)
        
        torchvision.utils.save_image(pics, f"{outdir}/all.png", nrow=8, padding=2, pad_value=255)
        print(f"Saved {num_samples} generated images to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate samples from a VAE with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")   
    parser.add_argument("--seed", type=int, default=0, help="Random seed")  
    parser.add_argument('--model', type=str, required=True, help='Model filename *.pth')
    parser.add_argument("--latent_dims", type=int, required=True, help="Number of latent dimensions (positive integer)")
    parser.add_argument("--max_filters", type=int, default=128, help="Maximum number of filters in the convolutional layers")
    parser.add_argument("--nsamples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for the generated samples')
    parser.add_argument('--nimg_channels', type=int, default=3, help='Number of image channels (1 for grayscale, 3 for RGB)')


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device('cpu')
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    if (not args.outdir.endswith("/")):
        args.outdir = args.outdir + "/"
    utils.ensure_folder_exists(args.outdir)
    generate(dev, args.model, args.latent_dims, args.nimg_channels, args.max_filters, args.nsamples, args.outdir)
