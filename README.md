# torch-mnist-vae
Variational autoencoder (VAE) implementation in PyTorch on mnist.  

# Requirements
* torch
* torchvision
* torchinfo
* numpy
* Pillow

See `requirements.txt`

# TODO

# Usage
```
$ python train.py -h
usage: Train a VAE on mnist with PyTorch. [-h] [--cpu] [--seed SEED] [--batchsize BATCHSIZE] [--epochs EPOCHS]
                                                            [--lr LR] [--dataset {CIFAR-10,CIFAR-100}]

options:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of Metal GPU acceleration
  --seed SEED           Random seed
  --batchsize BATCHSIZE
                        Batch size for training
  --epochs EPOCHS       Number of training epochs
  --lr LR               Learning rate
  --dataset {mnist}
                        Select the dataset to use (mnist)
```
