# Neural Network Training Framework

A comprehensive framework for training various neural network architectures with a focus on representation learning methods. This project supports different training approaches for both MNIST and CIFAR10 datasets.

## Features

- **Multiple Training Methods**:
  - Self-supervised learning (autoencoder)
  - Contrastive learning 
  - Classification-guided learning
  - Standard supervised classification

- **Supported Datasets**:
  - MNIST
  - CIFAR10

- **Visualization Tools**:
  - Training/validation metric plots
  - t-SNE visualization for latent spaces
  - Reconstruction visualization for autoencoders

- **Training Features**:
  - Model checkpointing
  - Learning rate scheduling
  - Data augmentation
  - Per-class accuracy tracking

## Project Structure

- `main.py`: Entry point for training
- `base_trainer.py`: Base class for all trainers
- `autoencoder_trainer.py`: Self-supervised autoencoder training
- `contrastive_trainer.py`: Contrastive learning implementation
- `classifier_trainer.py`: Supervised classifier training
- `classification_guided_trainer.py`: Joint encoder-classifier training
- `models.py`: Model architecture definitions
- `utils.py`: Utility functions and dataset setup

Results are saved under:

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