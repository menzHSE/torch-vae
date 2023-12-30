# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import utils
    
def get_loaders(dataset_name, img_size, batch_size, root="./data"):
    load_fn = None
    num_img_channels = 0
    if dataset_name == "mnist":
        load_fn = torchvision.datasets.MNIST
        num_img_channels = 1
    elif dataset_name == "cifar-10":
        load_fn = torchvision.datasets.CIFAR10
        num_img_channels = 3
    elif dataset_name == "cifar-100":
        load_fn = torchvision.datasets.CIFAR10
        num_img_channels = 3
    elif dataset_name == "celeb-a":
        load_fn = torchvision.datasets.CelebA
        num_img_channels = 3
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    train_loader, test_loader, classes_list = torchvision_load(dataset_name, batch_size, load_fn, img_size, root)
    return train_loader, test_loader, classes_list, num_img_channels 

def torchvision_load(dataset_name, batch_size, load_fn, img_size=(32,32), root="./data"):

    transform = transforms.Compose(        [
         transforms.Resize(img_size),  # resize the image to img_size pixels
         transforms.ToTensor()         # convert to tensor. This will also normalize pixels to 0-1
        ])

    # load train and test sets using torchvision
    if dataset_name == "celeb-a":
        tr   = load_fn(root=root, split="train", download=True, transform=transform)
        test = load_fn(root=root, split="test",  download=True, transform=transform)
        classes_list = None # could use "identity" attribute of the dataset
    elif dataset_name in ["cifar-100", "cifar-10", "mnist"]:
        tr   = load_fn(root=root, train=True,   download=True, transform=transform)
        test = load_fn(root=root, train=False,  download=True, transform=transform)
        classes_list = tr.classes
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
   
    # Data loaders
    train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=2)
    
    test_loader  = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, pin_memory=True, num_workers=2)
    

    
    return train_loader, test_loader, classes_list    


if __name__ == "__main__":

    batch_size = 32
    img_size = (64, 64)

    tr_loader, test_loader, classes_list, num_img_channels = get_loaders("mnist", img_size=img_size, batch_size=batch_size)

    B, C, H, W = batch_size, num_img_channels, img_size[0], img_size[1]   
    print(f"Batch size: {B}, Channels: {C}, Height: {H}, Width: {W}")

   
    images, labels = next(iter(tr_loader)) 
    assert images.shape == (B, C, H, W), "Wrong training set size"
    assert labels.shape == (B, ),        "Wrong training set size"


    images, labels = next(iter(test_loader))
    assert images.shape == (B, C, H, W), "Wrong training set size"
    assert labels.shape == (B, ),        "Wrong training set size"
   
    print(f"Classes : {classes_list}")

    # Save an image as a sanity check
        
    # Convert PyTorch tensor to numpy array and scale to 0-255
    img_data = (images[0].detach().cpu().numpy() * 255).astype(np.uint8)

    # Save the image using Pillow    
    utils.save_image(img_data, "/tmp/trainTmp.png")

    print("Dataset prepared successfully!")
