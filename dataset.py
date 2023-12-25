# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import os
import math
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

    

def mnist(batch_size, root="./data"):

    transform = transforms.Compose(        [
         transforms.Resize((32, 32)),  # resize the image to 32x32 pixels
         transforms.ToTensor()         # convert to tensor. This will also normalize pixels to 0-1
        ])

    # load train and test sets using torchvision
    tr   = torchvision.datasets.MNIST(root=root, train=True,   download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=root, train=False,  download=True, transform=transform)
   
    # Data loaders
    train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=2)
    
    test_loader  = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, pin_memory=True, num_workers=2)
    
    return train_loader, test_loader, tr.classes    


if __name__ == "__main__":

    batch_size = 32
    tr_loader, test_loader, classes = mnist(batch_size=batch_size)

    images, labels = next(iter(tr_loader)) 
    assert images.shape == (batch_size, 1, 28, 28), "Wrong training set size"
    assert labels.shape == (batch_size,), "Wrong training set size"


    images, labels = next(iter(test_loader))
    assert images.shape == (batch_size, 1, 28, 28), "Wrong training set size"
    assert labels.shape == (batch_size,), "Wrong training set size"
   
    print(classes)

    # Save an image as a sanity check
        
    # Normalize the image and convert to numpy array
    img_data = (images[0].numpy() * 255).astype(np.uint8)
    img_data = img_data.squeeze()

    # Save the image using Pillow
    img = Image.fromarray(img_data)
    img.save("/tmp/trainTmp.png")


    print("Dataset prepared successfully!")
