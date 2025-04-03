# Force non-interactive Agg backend - add this at the very top
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import MNISTEncoder, CIFAR10Encoder, MNISTAutoencoder, CIFAR10Autoencoder

def plot_tsne(model, dataloader, device, dataset_name=None):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to over over data for which you wish to compute projections
    device - cuda or cpu (as a string)
    dataset_name - Optional name of dataset for file naming
    '''
    model.eval()
    
    images_list = []
    labels_list = []
    latent_list = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            #approximate the latent space from data
            latent_vector = model(images)
            
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)
    
    # Determine dataset name for file paths
    ds_name = dataset_name if dataset_name else "unknown"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
    plt.colorbar(scatter)
    plt.title(f't-SNE of Latent Space ({ds_name})')
    plt.savefig(f'results/{ds_name}_latent_tsne.png')
    plt.close()
    
    #plot image domain tsne
    tsne_image = TSNE(n_components=2, random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)  
    plt.colorbar(scatter)
    plt.title(f't-SNE of Image Space ({ds_name})')
    plt.savefig(f'results/{ds_name}_image_tsne.png')
    plt.close()

def setup_datasets(args, model_type='encoder'):
    """
    Create and return datasets and dataloaders based on arguments.
    """
    # Determine dataset configuration based on dataset type
    if args.mnist:
        dataset_config = {
            'name': "MNIST",
            'img_size': 28,
            'is_mnist': True,
            'num_classes': 10,
            'dataset_class': datasets.MNIST,
            'mean': [0.5],
            'std': [0.5],
            'model_class': MNISTAutoencoder if model_type == 'autoencoder' else MNISTEncoder
        }
        # MNIST-specific augmentations
        if model_type == 'contrastive':
            augmentations = []
        elif model_type == 'autoencoder':
            # Augmentations for autoencoder
            augmentations = [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ]
        elif model_type == 'classification_guided':
            # Stronger augmentations for classification-guided training
            augmentations = [
                transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ]
        else:  
            # Basic augmentations for classifier training data
            augmentations = [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
            ]
    else:  # CIFAR10
        dataset_config = {
            'name': "CIFAR10",
            'img_size': 32,
            'is_mnist': False,
            'num_classes': 10,
            'dataset_class': datasets.CIFAR10,
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
            'model_class': CIFAR10Autoencoder if model_type == 'autoencoder' else CIFAR10Encoder
        }
        
        #CIFAR10 augmentations
        if model_type == 'contrastive':
            augmentations = []
        elif model_type == 'autoencoder':
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]
        elif model_type == 'classification_guided':
            augmentations = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2)
            ]
        else:  
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]

    # Create transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_config['mean'], std=dataset_config['std'])
    ])
    
    # For classification-guided training specifically
    if model_type == 'classification_guided':
        train_transform = transforms.Compose(
            augmentations + [
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_config['mean'], std=dataset_config['std']),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ]
        )
    else:
        train_transform = transforms.Compose(
            augmentations + [
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_config['mean'], std=dataset_config['std'])
            ]
        )
    
    # Create datasets
    train_dataset = dataset_config['dataset_class'](
        root=args.data_path, train=True, download=True, transform=train_transform
    )
    
    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    test_dataset = dataset_config['dataset_class'](
        root=args.data_path, train=False, download=True, transform=base_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model instance 
    model_constructor = dataset_config['model_class']
    model = model_constructor(args.latent_dim).to(args.device)
    
    setup = {
        'dataset_name': dataset_config['name'],
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_mnist': dataset_config['is_mnist'],
        'img_size': dataset_config['img_size'],
        'num_classes': dataset_config['num_classes']
    }
    
    # Add model and encoder based on model_type
    if model_type in ['autoencoder', 'contrastive', 'classification_guided']:
        setup['model'] = model
        if model_type in ['contrastive', 'classification_guided']:
            setup['encoder'] = model
    
    return setup

def get_result_dir(args, model_type, preTrained=False):
    """Get the appropriate results directory based on model type and dataset"""
    # Create the base results directory if it doesn't exist
    model_type = model_type.replace('-', '_')
    
    if not os.path.exists('results'):
        os.makedirs('results')
    # Determine dataset name from args.mnist flag
    dataset_name = 'mnist' if args.mnist else 'cifar10'
    if args.is_classifier and not preTrained:
        result_dir = os.path.join('results', 'classifier', args.encoder_type, dataset_name)
    else:
        result_dir = os.path.join('results', model_type, dataset_name)
    # Create the directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    return result_dir