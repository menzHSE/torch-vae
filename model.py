# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, num_latent_dims, device):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.device = device

        # we assume Bx1x32x32 input 

        # layers
        self.conv1 = nn.Conv2d   (1,        32, 3, stride=2, padding=1) # Output: 32x16x16
        self.conv2 = nn.Conv2d   (32,       64, 3, stride=2, padding=1) # Output: 64x8x8
        self.conv3 = nn.Conv2d   (64,      128, 3, stride=2, padding=1) # Output: 128x4x4
        # assuming 1x28x28 input image we get 128x4x4 of the conv layer stack     
        
        # normal distribution and KL divergence
        self.N      = torch.distributions.Normal(0, 1)  
        self.kl_div = 0  

        # linear mappings to mean and standard deviation
        # std-dev is directly outputted as but rather as a 
        # vector of log-variances. This is because the
        # standard deviation must be positive and the exp()
        # in forward ensures this. It might also be numerically
        # more stable.
        self.proj_mean  = nn.Linear(128*4*4, num_latent_dims)
        self.proj_sigma = nn.Linear(128*4*4, num_latent_dims) 
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        mu    = self.proj_mean(x)
        sigma = torch.exp(self.proj_sigma(x))

        # sample from the distribution
        sampled_z = self.N.sample(mu.shape).to(mu.device)
        z = mu + sigma * sampled_z
        # compute KL divergence
        self.kl_div = torch.sum(sigma**2 + mu**2 - torch.log(sigma) - 1/2)
        return z # return latent vector
        
class Decoder(nn.Module):
    """A convolutional decoder """

    def __init__(self, num_latent_dims):
        super().__init__()
        self.num_latent_dims = num_latent_dims

        self.lin1  = nn.Linear(num_latent_dims, 128*4*4) # Output: 128x4x4
        self.conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # Output: 64x8x8
        self.conv2 = nn.ConvTranspose2d(64,  32, 3, stride=2, padding=1, output_padding=1)  # Output: 32x16x16
        self.conv3 = nn.ConvTranspose2d(32,   1, 3, stride=2, padding=1, output_padding=1)  # Output: 1x32x32
        

    def forward(self, z):
        # unflatten the latent vector
        z = self.lin1(z)
        z = z.view(-1, 128, 4, 4)
        z = F.leaky_relu(self.conv1(z))
        z = F.leaky_relu(self.conv2(z))
        z = F.sigmoid(self.conv3(z)) 
        return z
        

        

class VAE(nn.Module):
    """A convolutional Variational Autoencoder """

    def __init__(self, num_latent_dims, device):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.encoder = Encoder(num_latent_dims, device).to(device)
        self.decoder = Decoder(num_latent_dims).to(device)
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
        self.load_state_dict(torch.load(fname))
        self.eval()

def vae_loss_fn(x, x_recon, kl_div):
    """The loss function for the VAE. It is a combination of the reconstruction loss
    and the KL divergence between the latent distribution and the standard normal distribution.
    """

    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # Total loss
    loss = recon_loss + kl_div
    return loss

