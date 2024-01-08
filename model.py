# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    """
    A convolutional variational encoder. We do not map the input image
    deterministically to a latent vector. Instead, we map the input to
    a probability distribution in latent space and sample a latent vector
    fron that distribution. In this example, we linearly map the input
    image to a mean vector and a vector of standard deviations that
    parameterize a normal distribution.

    We can then sample from this distribution to generate a new image. Also,
    we can add an auxiliary loss to the network that forces the distribution
    to be close to a standard normal distribution. We use the KL divergence
    between the two distributions as this auxiliary loss.
    """

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, device):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.device = device

        # we assume B x #img_channels x 64 x 64 input
        # Todo: add input shape attribute to the model to make it more flexible

        # C x H x W
        img_input_shape = (num_img_channels, 64, 64)

        # layers (with max_num_filters=128)

        num_filters_1 = max_num_filters // 4
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters

        # print(f"Encoder: ")
        # print(f"  num_filters_1={num_filters_1}")
        # print(f"  num_filters_2={num_filters_2}")
        # print(f"  num_filters_3={num_filters_3}")

        # Output: num_filters_1 x 32 x 32
        self.conv1 = nn.Conv2d(num_img_channels, num_filters_1, 3, stride=2, padding=1)
        # Output: num_filters_2 x 16 x 16
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, 3, stride=2, padding=1)
        # Output: num_filters_3 x 8 x 8
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, 3, stride=2, padding=1)

        # Shortcuts
        self.shortcut2 = nn.Conv2d(num_filters_1, num_filters_2, 1, stride=2, padding=0)
        self.shortcut3 = nn.Conv2d(num_filters_2, num_filters_3, 1, stride=2, padding=0)

        # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(num_filters_1)
        self.bn2 = nn.BatchNorm2d(num_filters_2)
        self.bn3 = nn.BatchNorm2d(num_filters_3)

        # linear mappings to mean and standard deviation

        # std-dev is not directly outputted but rather as a
        # vector of log-variances. This is because the
        # standard deviation must be positive and the exp()
        # in forward ensures this. It might also be numerically
        # more stable.

        # divide the last two dimensions by 8 because of the 3 strided convolutions
        output_shape = [num_filters_3] + [
            dimension // 8 for dimension in img_input_shape[1:]
        ]
        flattened_dim = math.prod(output_shape)

        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma = torch.exp(
            logvar * 0.5
        )  # Ensure this is the std deviation, not variance

        # Generate a tensor of random values from a normal distribution
        eps = torch.randn_like(sigma)

        # Perform the reparametrization step ...
        # This allows us to backpropagate through it, which we could not do,
        # if we had just sampled from a normal distribution with mean mu and
        # standard deviation sigma. The multiplication with sigma and addition
        # of mu is just a linear transformation of the random values from the
        # normal distribution. The result is a random value from the distribution
        # with mean mu and standard deviation sigma. Backpropagation is possible
        # because the gradients of the random values are just 1 and the gradients
        # of the linear transformation are just the weights of the linear transformation.
        z = eps.mul(sigma).add_(mu)

        # compute KL divergence
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        self.kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z  # return latent vector


class Decoder(nn.Module):
    """A convolutional decoder"""

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.input_shape = None

        # decoder layers
        num_filters_1 = max_num_filters
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters // 4

        # print(f"Decoder: ")
        # print(f"  num_filters_1={num_filters_1}")
        # print(f"  num_filters_2={num_filters_2}")
        # print(f"  num_filters_3={num_filters_3}")

        # C x H x W
        img_output_shape = (num_img_channels, 64, 64)

        # divide the last two dimensions by 8 because of the 3 strided convolutions
        self.input_shape = [num_filters_1] + [
            dimension // 8 for dimension in img_output_shape[1:]
        ]
        flattened_dim = math.prod(self.input_shape)

        # Output: flattened_dim
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)
        # Output: num_filters_2 x 16 x 16
        self.conv1 = nn.ConvTranspose2d(
            num_filters_1, num_filters_2, 3, stride=2, padding=1, output_padding=1
        )
        # Output: num_filters_1 x 32 x 32
        self.conv2 = nn.ConvTranspose2d(
            num_filters_2, num_filters_3, 3, stride=2, padding=1, output_padding=1
        )
        # Output: #img_channels x 64 x 64
        self.conv3 = nn.ConvTranspose2d(
            num_filters_3, num_img_channels, 3, stride=2, padding=1, output_padding=1
        )

        # Shortcuts
        self.shortcut1 = nn.ConvTranspose2d(
            num_filters_1, num_filters_2, 1, stride=2, padding=0, output_padding=1
        )
        self.shortcut2 = nn.ConvTranspose2d(
            num_filters_2, num_filters_3, 1, stride=2, padding=0, output_padding=1
        )

        # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(num_filters_2)
        self.bn2 = nn.BatchNorm2d(num_filters_3)

    def forward(self, z):
        # unflatten the latent vector
        x = self.lin1(z)
        x = x.view(-1, self.max_num_filters, self.input_shape[-2], self.input_shape[-1])
        # poor man's ResNet -> skip connections
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.sigmoid(self.conv3(x))  # sigmoid to ensure pixel values are in [0,1]
        return x


class VAE(nn.Module):
    """A convolutional Variational Autoencoder"""

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, device):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.device = device
        self.encoder = Encoder(
            num_latent_dims, num_img_channels, max_num_filters, device
        )
        self.decoder = Decoder(num_latent_dims, num_img_channels, max_num_filters)
        self.kl_div = 0

    # forward pass of the data "x"
    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        self.kl_div = self.encoder.kl_div
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save(self, fname):
        utils.ensure_folder_exists(fname)
        # save the model
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location=self.device))
        self.eval()
