# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

        # layers
        # Output: 16x16x16
        self.conv1 = nn.Conv2d   (num_img_channels, 16, 3, stride=2, padding=1) 
        # Output: 32x8x8
        self.conv2 = nn.Conv2d   (16,               32, 3, stride=2, padding=1) 
        # Output: 64x4x4
        self.conv3 = nn.Conv2d   (32,               64, 3, stride=2, padding=1) 
                       

        # linear mappings to mean and standard deviation
        # std-dev is directly outputted as but rather as a 
        # vector of log-variances. This is because the
        # standard deviation must be positive and the exp()
        # in forward ensures this. It might also be numerically
        # more stable.
        self.proj_mu      = nn.Linear(64*4*4, num_latent_dims)
        self.proj_log_var = nn.Linear(64*4*4, num_latent_dims) 
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
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
        self.lin1  = nn.Linear(num_latent_dims, 64*4*4) 
        # Output: 32x8x8
        self.conv1 = nn.ConvTranspose2d(64, 32,               3, stride=2, padding=1, output_padding=1)  
        # Output: 16x16x16
        self.conv2 = nn.ConvTranspose2d(32,  16,               3, stride=2, padding=1, output_padding=1)  
        # Output: #img_channelsx32x32
        self.conv3 = nn.ConvTranspose2d(16,  num_img_channels, 3, stride=2, padding=1, output_padding=1)  
        

    def forward(self, z):
        # unflatten the latent vector
        z = self.lin1(z)
        z = z.view(-1, 64, 4, 4)
        z = F.leaky_relu(self.conv1(z))
        z = F.leaky_relu(self.conv2(z))
        z = F.sigmoid(self.conv3(z)) # sigmoid to ensure pixel values are in [0,1]
        return z
        

        
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
        # Extract the directory path from the file name
        dir_path = os.path.dirname(fname)

        # Check if the directory path is not empty
        if dir_path:
            # Check if the directory exists, and create it if it does not
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        # save the model
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location=self.device))
        self.eval()
