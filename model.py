# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils


class Encoder(nn.Module):
    """
    A convolutional variational encoder. We do not map the input image 
    deterministically to a latent vector. Instead, we map the input to 
    a probability distribution in latent space and sample a latent vector
    fron that distribution. In this example,we linearly map the input
    image to a mean vector and a vector of standard deviations that 
    parameterize a normal distribution.

    We can then sample from this distribution to generate a new image. Also,
    we can add an auxiliary loss to the network that forces the distribution
    to be close to a standard normal distribution. We use the KL divergence
    between the two distributions as this auxiliary loss.     
    """

    def __init__(self, num_latent_dims, num_img_channels, device):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.device = device

        # we assume B x #img_channels x 32 x 32 input 
        # Todo: add input shape attribute to the model to make it more flexible

        # layers
        # Output: 16x16x16
        self.conv1 = nn.Conv2d   (num_img_channels,  64, 3, stride=2, padding=1) 
        # Output: 32x8x8
        self.conv2 = nn.Conv2d   (64,               128, 3, stride=2, padding=1) 
        # Output: 64x4x4
        self.conv3 = nn.Conv2d   (128,              256, 3, stride=2, padding=1) 
        
        # Shortcuts
        self.shortcut2 = nn.Conv2d(64,  128, 1, stride=2, padding=0)
        self.shortcut3 = nn.Conv2d(128, 256, 1, stride=2, padding=0)

        # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
                       

        # linear mappings to mean and standard deviation
        # std-dev is directly outputted as but rather as a 
        # vector of log-variances. This is because the
        # standard deviation must be positive and the exp()
        # in forward ensures this. It might also be numerically
        # more stable.
        self.proj_mu      = nn.Linear(256*4*4, num_latent_dims)
        self.proj_log_var = nn.Linear(256*4*4, num_latent_dims) 
        
    def forward(self, x):

        # poor man's ResNet -> skip connections
        
        x =                     F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.shortcut2(x) + F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.shortcut3(x) + F.leaky_relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        mu     = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma  = torch.exp(logvar * 0.5)  # Ensure this is the std deviation, not variance

        # Generate a tensor of random values from a normal distribution
        eps = torch.randn_like(sigma) 

        # Perform the reparametrization step
        z = eps.mul(sigma).add_(mu)

        # compute KL divergence
         # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        self.kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z # return latent vector
        
class Decoder(nn.Module):
    """A convolutional decoder """

    def __init__(self, num_latent_dims, num_img_channels):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels

        # Output: 64x4x4
        self.lin1  = nn.Linear(num_latent_dims, 256*4*4) 
        # Output: 32x8x8
        self.conv1 = nn.ConvTranspose2d(256,  128,              3, stride=2, padding=1, output_padding=1)  
        # Output: 16x16x16
        self.conv2 = nn.ConvTranspose2d(128,  64,               3, stride=2, padding=1, output_padding=1)  
        # Output: #img_channelsx32x32
        self.conv3 = nn.ConvTranspose2d( 64,  num_img_channels, 3, stride=2, padding=1, output_padding=1)  

         # Shortcuts
        self.shortcut1 = nn.ConvTranspose2d(256, 128, 1, stride=2, padding=0, output_padding=1)
        self.shortcut2 = nn.ConvTranspose2d(128, 64, 1, stride=2, padding=0, output_padding=1)

         # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
     

    def forward(self, z):
        # unflatten the latent vector
        x = self.lin1(z)
        x = x.view(-1, 256, 4, 4)
        # poor man's ResNet -> skip connections
        x = self.shortcut1(x) + F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.shortcut2(x) + F.leaky_relu(self.bn2(self.conv2(x)))
        x =                     F.sigmoid(            self.conv3(x)) # sigmoid to ensure pixel values are in [0,1]
        return x
        

        
class VAE(nn.Module):
    """A convolutional Variational Autoencoder """

    def __init__(self, num_latent_dims, num_img_channels, device):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.device=device
        self.encoder = Encoder(num_latent_dims, num_img_channels, device)
        self.decoder = Decoder(num_latent_dims, num_img_channels)
        self.kl_div  = 0

    # forward pass of the data "x"
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        self.kl_div = self.encoder.kl_div
        return x
    
 
    def save(self, fname):

        utils.ensure_folder_exists(fname)
        # save the model
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location=self.device))
        self.eval()
