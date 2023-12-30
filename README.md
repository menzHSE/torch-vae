# torch-vae

Author: Markus Enzweiler, markus.enzweiler@hs-esslingen.de

Convolutional variational autoencoder (VAE) implementation in PyTorch. Supported datasets include MNIST, CIFAR-10/100 and CelebA. See https://github.com/menzHSE/cv-ml-lecture-notebooks for interactive Jupyter notebooks using this pacakage with additional explanations and visualizations. 

A good overview of variational autoencoders is given in https://arxiv.org/abs/1906.02691 and https://arxiv.org/abs/1312.6114.


## Requirements
* torch
* torchvision
* torchinfo
* numpy
* Pillow

See `requirements.txt`


## Usage

### Model Training

Pretrained models for all datasets are available in the ```models``` directory. The models carry information of the maximum number of filters in the conv layers (```--max_filters```) and the number of latent dimensions (```--latent_dims```) in their filename. These models use three conv layers with 32/64/128 features (and corresponding transposed conv layers in the decoder) and 64 latent dimensions. To train a VAE model use ```python train.py```. 

```
python train.py  -h
usage: Train a VAE with PyTorch. [-h] [--cpu] [--seed SEED] [--batchsize BATCHSIZE] [--max_filters MAX_FILTERS]
                                 [--epochs EPOCHS] [--lr LR] [--dataset {mnist,cifar-10,cifar-100,celeb-a}] --latent_dims
                                 LATENT_DIMS

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of GPU (cuda/mps) acceleration
  --seed SEED           Random seed
  --batchsize BATCHSIZE
                        Batch size for training
  --max_filters MAX_FILTERS
                        Maximum number of filters in the convolutional layers
  --epochs EPOCHS       Number of training epochs
  --lr LR               Learning rate
  --dataset {mnist,cifar-10,cifar-100,celeb-a}
                        Select the dataset to use (mnist, cifar-10, cifar-100, celeb-a)
  --latent_dims LATENT_DIMS
                        Number of latent dimensions (positive integer)
```
**Example**

```python train.py --batchsize=128 --epochs=100 --dataset=celeb-a --latent_dims=64```

### Reconstruction of Training / Test Data

Datasets can be reconstructed using ```python reconstruct.py```. Images depicting original and reconstructed data samples are written to the folder specified by ```--outdir```.

``` 
python reconstruct.py  -h
usage: Reconstruct data samples using a VAE with PyTorch. [-h] [--cpu] --model MODEL [--rec_testdata] [--dataset {mnist,cifar-10,cifar-100,celeb-a}]
                                                          --latent_dims LATENT_DIMS [--max_filters MAX_FILTERS] --outdir OUTDIR

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of GPU (cuda/mps) acceleration
  --model MODEL         Model filename *.pth
  --rec_testdata        Reconstruct test split instead of training split
  --dataset {mnist,cifar-10,cifar-100,celeb-a}
                        Select the dataset to use (mnist, cifar-10, cifar-100, celeb-a)
  --latent_dims LATENT_DIMS
                        Number of latent dimensions (positive integer)
  --max_filters MAX_FILTERS
                        Maximum number of filters in the convolutional layers
  --outdir OUTDIR       Output directory for the generated samples
```


#### Examples

**Reconstructing MNIST**

```python reconstruct.py --model=models/mnist/vae_filters_0128_dims_0064.pth  --dataset=mnist  --latent_dims=64 --outdir=reconstructions/mnist```

![MNIST Reconstructions](docs/images/rec_mnist.jpg)


**Reconstructing CelebA**

```python reconstruct.py --model=models/celeb-a/vae_filters_0128_dims_0064.pth  --dataset=celeb-a --latent_dims=64 --outdir=reconstructions/celeb-a```

![CelebA Reconstructions](docs/images/rec_celeb-a.jpg)


### Generating Samples from the Model

The variational autoencoders are trained in a way that the distribution in latent space resembles a normal distribution (there is a special loss term for that). To generate samples from the variational autoencoder, we can sample a random normally distributed latent vector and have the decoder generate an image from that. Use ```python generate.py``` to generate random samples. 


``` 
python generate.py -h
usage: Generate samples from a VAE with PyTorch. [-h] [--cpu] [--seed SEED] --model MODEL --latent_dims LATENT_DIMS [--max_filters MAX_FILTERS]
                                                 [--nsamples NSAMPLES] --outdir OUTDIR [--nimg_channels NIMG_CHANNELS]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of GPU (cuda/mps) acceleration
  --seed SEED           Random seed
  --model MODEL         Model filename *.pth
  --latent_dims LATENT_DIMS
                        Number of latent dimensions (positive integer)
  --max_filters MAX_FILTERS
                        Maximum number of filters in the convolutional layers
  --nsamples NSAMPLES   Number of samples to generate
  --outdir OUTDIR       Output directory for the generated samples
  --nimg_channels NIMG_CHANNELS
                        Number of image channels (1 for grayscale, 3 for RGB)
```

#### Examples

**Sample from the VAE models trained on MNIST**

```python generate.py  --model=models/mnist/vae_filters_0128_dims_0064.pth  --latent_dims=64 --outdir=./samples/mnist --nimg_channels=1 --nsamples=64``` 

![MNIST Samples](docs/images/samples_mnist.jpg)


**Sample from the VAE models trained on CelebA**

```python generate.py  --model=models/celeb-a/vae_filters_0128_dims_0064.pth  --latent_dims=64 --outdir=./samples/celeb-a/ --nimg_channels=3 --nsamples=64```

![CelebA Samples](docs/images/samples_celeb-a.jpg)

