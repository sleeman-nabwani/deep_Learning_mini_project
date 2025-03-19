# Self-Supervised Autoencoder Implementation

This repository contains the implementation of a self-supervised autoencoder for image classification using MNIST and CIFAR10 datasets.

## Project Structure

- `code/models.py`: Contains the encoder, decoder, and classifier model architectures for both MNIST and CIFAR10 datasets.
- `code/train_autoencoder.py`: Contains the code for training the autoencoder in self-supervised mode.
- `code/train_classifier.py`: Contains the code for training a classifier on top of the pre-trained encoder.
- `code/main.py`: Main script to run training for either autoencoder or classifier.
- `code/utils.py`: Contains utility functions for visualization.

## Requirements

The required packages are listed in `code/environment.yml`.

## Usage

### 1. Self-Supervised Autoencoder Training

To train the autoencoder in self-supervised mode for MNIST:

```bash
python code/main.py --mnist --self-supervised --data-path /datasets/cv_datasets/data --batch-size 64 --epochs 20
```

For CIFAR10:

```bash
python code/main.py --self-supervised --data-path /datasets/cv_datasets/data --batch-size 64 --epochs 20
```

### 2. Training a Classifier on top of the Pre-trained Encoder

Make sure you've trained the autoencoder first, then:

For MNIST:

```bash
python code/main.py --mnist --train-classifier --data-path /datasets/cv_datasets/data --batch-size 64 --epochs 20
```

For CIFAR10:

```bash
python code/main.py --train-classifier --data-path /datasets/cv_datasets/data --batch-size 64 --epochs 20
```

### 3. Training Both in Sequence

To train both the autoencoder and classifier in sequence:

For MNIST:

```bash
python code/main.py --mnist --self-supervised --train-classifier --data-path /datasets/cv_datasets/data --batch-size 64 --epochs 20
```

For CIFAR10:

```bash
python code/main.py --self-supervised --train-classifier --data-path /datasets/cv_datasets/data --batch-size 64 --epochs 20
```

## Results

The training results are saved in the `results` directory:
- Model checkpoints (`.pth` files)
- Training and validation loss/accuracy curves
- Sample reconstructions
- t-SNE plots of the latent space

## Notes

- The autoencoder projects images to a 128-dimensional latent space.
- The encoder is frozen when training the classifier.
- All models are trained with Adam optimizer with a learning rate of 1e-3.