import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from models import MNISTAutoencoder, CIFAR10Autoencoder
from utils import plot_tsne
import torch.nn.functional as F

def train_autoencoder(args):
    # Set device
    device = args.device
    
    # Set up transforms with data augmentation for CIFAR10
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
        model = MNISTAutoencoder(args.latent_dim).to(device)
        dataset_name = "MNIST"
        img_size = 28
    else:
        # Enhanced data augmentation for CIFAR10, but more conservative for autoencoder
        train_transform = transforms.Compose([
            # Only horizontal flip for autoencoder to avoid extreme transformations
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # No augmentation for validation/test
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        model = CIFAR10Autoencoder(args.latent_dim).to(device)
        dataset_name = "CIFAR10"
        img_size = 32
    
    # Split training dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    os.makedirs('results', exist_ok=True)
    train_losses = []
    val_losses = []
    train_recon_scores = []
    val_recon_scores = []
    best_val_loss = float('inf')
    
    print(f"[AUTOENCODER] Starting training for {dataset_name} dataset with latent dim={args.latent_dim}")
    print(f"[AUTOENCODER] Training on {device} with batch size {args.batch_size} for {args.epochs} epochs")
    print(f"[AUTOENCODER] Image size: {img_size}x{img_size}")
    print(f"[AUTOENCODER] Using data augmentation: {not args.mnist}")
    
    # Verify model output dimensions with a small test batch
    with torch.no_grad():
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        sample_output = model(sample_batch)
        if sample_output.shape != sample_batch.shape:
            print(f"[WARNING] Model output shape {sample_output.shape} doesn't match input shape {sample_batch.shape}")
            print("[WARNING] This may cause issues with the loss calculation")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon_score = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Verify dimensions match
            if output.shape != data.shape:
                print(f"[WARNING] Batch {batch_idx}: Output shape {output.shape} doesn't match input shape {data.shape}")
                if output.shape[2:] != data.shape[2:]:
                    print(f"[AUTOENCODER] Resizing output to match input dimensions")
                    output = F.interpolate(output, size=data.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(output, data)
            
            # Calculate reconstruction score (1 - normalized MSE as a percentage)
            # This gives us a "reconstruction accuracy" between 0-100%
            with torch.no_grad():
                mse = F.mse_loss(output, data, reduction='sum').item() / (data.size(0) * data.numel() / data.size(0))
                recon_score = 100 * (1 - min(mse, 1.0))  # Cap at 0% if MSE > 1
                train_recon_score += recon_score
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"[AUTOENCODER] {dataset_name} - Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, Recon Score: {recon_score:.2f}%")
        
        train_loss /= len(train_loader)
        train_recon_score /= len(train_loader)
        train_losses.append(train_loss)
        train_recon_scores.append(train_recon_score)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_score = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                output = model(data)
                
                # Verify dimensions match
                if output.shape != data.shape:
                    if output.shape[2:] != data.shape[2:]:
                        output = F.interpolate(output, size=data.shape[2:], mode='bilinear', align_corners=False)
                
                loss = criterion(output, data)
                val_loss += loss.item()
                
                # Calculate reconstruction score for validation
                mse = F.mse_loss(output, data, reduction='sum').item() / (data.size(0) * data.numel() / data.size(0))
                recon_score = 100 * (1 - min(mse, 1.0))
                val_recon_score += recon_score
                
        val_loss /= len(val_loader)
        val_recon_score /= len(val_loader)
        val_losses.append(val_loss)
        val_recon_scores.append(val_recon_score)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f"[AUTOENCODER] {dataset_name} - Epoch: {epoch+1}/{args.epochs}, "
              f"Train Loss: {train_loss:.6f}, Train Recon Score: {train_recon_score:.2f}%, "
              f"Val Loss: {val_loss:.6f}, Val Recon Score: {val_recon_score:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = 'mnist_autoencoder.pth' if args.mnist else 'cifar10_autoencoder.pth'
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_recon_score': val_recon_score
            }, os.path.join('results', model_name))
            print(f"[AUTOENCODER] {dataset_name} - Saved model checkpoint with validation loss: {val_loss:.6f} and recon score: {val_recon_score:.2f}%")
        
        # Visualize reconstructions every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            visualize_reconstructions(model, test_loader, device, epoch, args.mnist)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} Autoencoder - Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_recon_scores, label='Training Recon Score')
    plt.plot(val_recon_scores, label='Validation Recon Score')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Score (%)')
    plt.title(f'{dataset_name} Autoencoder - Reconstruction Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/{'mnist' if args.mnist else 'cifar10'}_ae_curves.png")
    
    # Generate t-SNE plot for the latent space
    plot_tsne(model.encoder, test_loader, device, dataset_name=dataset_name.lower())
    
    print(f"[AUTOENCODER] {dataset_name} - Training completed!")
    print(f"[AUTOENCODER] {dataset_name} - Final validation loss: {val_losses[-1]:.6f}, Recon score: {val_recon_scores[-1]:.2f}%")
    
    return model

def visualize_reconstructions(model, dataloader, device, epoch, is_mnist):
    """Visualize original images and their reconstructions"""
    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        data, _ = next(iter(dataloader))
        data = data.to(device)
        
        # Get reconstructions
        reconstructions = model(data)
        
        # Move to CPU for visualization
        data = data.cpu()
        reconstructions = reconstructions.cpu()
        
        # Create figure
        n = min(8, data.size(0))
        plt.figure(figsize=(12, 6))
        
        # Plot original images
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            img = data[i].detach()
            if is_mnist:
                img = img.squeeze()
            else:
                img = img.permute(1, 2, 0)
                img = (img * 0.5) + 0.5  # Denormalize
            plt.imshow(img, cmap='gray' if is_mnist else None)
            plt.title("Original")
            plt.axis('off')
        
        # Plot reconstructions
        for i in range(n):
            ax = plt.subplot(2, n, n + i + 1)
            img = reconstructions[i].detach()
            if is_mnist:
                img = img.squeeze()
            else:
                img = img.permute(1, 2, 0)
                img = (img * 0.5) + 0.5  # Denormalize
            plt.imshow(img, cmap='gray' if is_mnist else None)
            plt.title("Reconstructed")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/{'mnist' if is_mnist else 'cifar10'}_reconstructions_epoch_{epoch+1}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False, help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_autoencoder(args) 