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
    # Initialize augmentations as an empty list by default
    augmentations = []
    
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
        # MNIST augmentations
        augmentations = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ]
    else:
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
        
        # CIFAR10 augmentations
        if model_type == 'autoencoder':
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
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
    
    train_transform = transforms.Compose(
        augmentations + [
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_config['mean'], std=dataset_config['std'])
        ]
    )
    
    # Create datasets and loaders
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
    
    # Create model
    model = dataset_config['model_class'](args.latent_dim).to(args.device)
    
    # Construct the return dictionary
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
    if model_type in ['autoencoder', 'encoder', 'classifier']:
        setup['model'] = model
        if model_type in ['encoder', 'classifier']:
            setup['encoder'] = model
    
    return setup

def get_result_dir(args, training_type):
    """
    Create and return an appropriate result directory based on 
    training type and dataset.
    
    Args:
        args: Command line arguments
        training_type: 'self_supervised' or 'classification_guided'
        
    Returns:
        str: Path to the result directory
    """
    dataset_name = "mnist" if args.mnist else "cifar10"
    result_dir = os.path.join('results', training_type, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir