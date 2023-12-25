# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import torch

# Check the devices that we have available and prefer CUDA over MPS and CPU
def autoselectDevice(verbose=0):

    # default: CPU
    device = torch.device('cpu')

    if torch.cuda.is_available():
        # CUDA
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # MPS (acceleration on Apple silicon M1 / M2 chips)
        device = torch.device('mps')

    if verbose:
        print('Using device:', device)

    # Additional Info when using cuda
    if verbose and device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    return device

