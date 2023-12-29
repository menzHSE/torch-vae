# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a convolutional variational autoencoder for in PyTorch

import argparse
import time
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import dataset
import model
import trainer
import device
import loss


# Train the CNN
def train(device, batch_size, num_epochs, learning_rate, dataset_name, num_latent_dims, max_num_filters):

    # Image size
    img_size = (64, 64)

    # get the data
    train_loader, _, _, num_img_channels = dataset.get_loaders(dataset_name, img_size, batch_size)

    # Instantiate the VAE
    vae = model.VAE(num_latent_dims, num_img_channels, max_num_filters, device=device)

    # print summary and correctly flush the stream
    model_stats = summary(vae, input_size=(1, num_img_channels, img_size[0], img_size[1]), row_settings=["var_names"])
    print("", flush=True)
    time.sleep(1)

    # Loss
    loss_fn = loss.mse_kl_loss

    # Optimizer
    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate)

    # Train the network
    fname_save_every_epoch = f"models/{dataset_name}/vae_filters_{vae.max_num_filters:04d}_dims_{vae.num_latent_dims:04d}"
    trainer_obj = trainer.Trainer(vae, loss_fn, optimizer, device, fname_save_every_epoch, log_level=logging.INFO)
    trainer_obj.train(train_loader, None, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a VAE with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_filters", type=int, default=128, help="Maximum number of filters in the convolutional layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Max number of filters: {args.max_filters}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of latent dimensions: {args.latent_dims}")

    train(dev, args.batchsize, args.epochs, args.lr, args.dataset, args.latent_dims, args.max_filters)
