import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from models import MNISTEncoder, CIFAR10Encoder, Classifier
from utils import plot_tsne

def train_classifier(args):
    # Set device
    device = args.device
    
    # Set up transforms
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
        encoder = MNISTEncoder(args.latent_dim).to(device)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        encoder = CIFAR10Encoder(args.latent_dim).to(device)
    
    # Load pre-trained encoder
    model_name = 'mnist_autoencoder.pth' if args.mnist else 'cifar10_autoencoder.pth'
    checkpoint = torch.load(os.path.join('results', model_name), map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    
    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    encoder.eval()  # Set encoder to evaluation mode
    
    # Create classifier
    classifier = Classifier(args.latent_dim, 10).to(device)
    
    # Split training dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Training loop
    os.makedirs('results', exist_ok=True)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print(f"Starting training classifier for {'MNIST' if args.mnist else 'CIFAR10'}...")
    
    for epoch in range(args.epochs):
        # Training
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Encode data with the frozen encoder
            with torch.no_grad():
                encoded_data = encoder(data)
            
            # Forward pass through classifier
            optimizer.zero_grad()
            outputs = classifier(encoded_data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}, Acc: {100.*correct/total:.2f}%")
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Encode data with the frozen encoder
                encoded_data = encoder(data)
                
                outputs = classifier(encoded_data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch: {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
        
        # Save model if validation accuracy has increased
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = 'mnist_classifier.pth' if args.mnist else 'cifar10_classifier.pth'
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss
            }, os.path.join('results', model_name))
            print(f"Saved model checkpoint with validation accuracy: {val_acc:.2f}%")
    
    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/{'mnist' if args.mnist else 'cifar10'}_classifier_curves.png")
    
    # Evaluate on test set
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Encode data with the frozen encoder
            encoded_data = encoder(data)
            
            outputs = classifier(encoded_data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Generate t-SNE plot for the latent space
    plot_tsne(encoder, test_loader, device)
    
    print("Training completed!")

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
    
    train_classifier(args) 