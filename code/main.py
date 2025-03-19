import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
import os
from models import MNISTAutoencoder, CIFAR10Autoencoder
import train_autoencoder
import train_classifier

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    parser.add_argument('--train-classifier', action='store_true', default=False,
                        help='Whether to train a classifier on top of the pretrained encoder')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Select appropriate transform based on dataset
    dataset_name = "MNIST" if args.mnist else "CIFAR10"
    
    print(f"=== {dataset_name} AUTOENCODER TRAINING ===")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Data path: {args.data_path}")
    print(f"Self-supervised mode: {args.self_supervised}")
    print(f"Train classifier: {args.train_classifier}")
    print("="*40)
    
    # Select appropriate transform based on dataset
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    # First train the autoencoder in self-supervised mode
    if args.self_supervised:
        print("\n=== STARTING AUTOENCODER TRAINING ===")
        autoencoder = train_autoencoder.train_autoencoder(args)
        print("=== AUTOENCODER TRAINING COMPLETE ===\n")
        
    # Then train the classifier on top of the pretrained encoder if requested
    if args.train_classifier:
        print("\n=== STARTING CLASSIFIER TRAINING ===")
        classifier = train_classifier.train_classifier(args)
        print("=== CLASSIFIER TRAINING COMPLETE ===\n")
    
    print(f"\n=== ALL TRAINING COMPLETED FOR {dataset_name} ===")
    if args.self_supervised and args.train_classifier:
        print("Both autoencoder and classifier were trained successfully.")
    elif args.self_supervised:
        print("Only autoencoder was trained. Run with --train-classifier to train a classifier.")
    elif args.train_classifier:
        print("Only classifier was trained using a pre-trained encoder.")

