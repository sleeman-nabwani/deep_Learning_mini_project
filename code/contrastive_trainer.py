import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from base_trainer import BaseTrainer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torchvision import transforms
import math
from utils import plot_tsne

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) from SimCLR
    
    This can be used as a drop-in criterion like nn.L1Loss or nn.CrossEntropyLoss
    """
    def __init__(self, temperature=0.1, hard_negative_weight=0.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
    
    def forward(self, z1, z2):
        """
        Args:
            z1: First batch of projections [batch_size, projection_dim]
            z2: Second batch of projections [batch_size, projection_dim]
        """
        batch_size = z1.shape[0]
        device = z1.device
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)   
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                              representations.unsqueeze(0), 
                                              dim=2)
        # Remove diagonal entries (self-similarity)
        sim_i_j = torch.diag(similarity_matrix, batch_size) 
        sim_j_i = torch.diag(similarity_matrix, -batch_size) 
        #similarities between augmented pairs
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Create mask to identify the negative samples
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=device) 
        #isolate only the negatives
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)    
        # Compute logits: positive / temperature
        logits = torch.cat([positives.unsqueeze(1) / self.temperature, 
                          negatives / self.temperature], dim=1)
        # Labels always point to the positives (index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        return loss


class ContrastiveTrainer(BaseTrainer):
    """Trainer for contrastive learning to structure latent space"""
    
    def __init__(self, args, setup):
        super().__init__(args, 'contrastive')
        self.encoder = setup.get('model') or setup.get('encoder')
        self.dataset_name = setup['dataset_name']
        self.train_loader = setup['train_loader']
        self.val_loader = setup['val_loader']
        self.test_loader = setup['test_loader']
        self.is_mnist = setup['is_mnist']
        
        #projection head 
        self.projection_head = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        ).to(args.device)
        
        self.temperature = 0.1 
        self.criterion = NTXentLoss(temperature=self.temperature)
        
        # Optimizer for both encoder and projection head
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=3e-4,
            weight_decay=1e-4
        )
        
        total_steps = len(self.train_loader) * self.epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=3e-3,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
        
        # File naming
        self.model_save_path = f'{self.dataset_name.lower()}_contrastive.pth'
        
        # Define augmentations for contrastive pairs
        self.augment = self._get_contrastive_augmentations()
    
    def _get_contrastive_augmentations(self):
        """augmentations for contrastive learning"""
        if self.is_mnist:
            # MNIST-specific augmentations
            return transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
                transforms.RandomAffine(
                    degrees=30,                
                    translate=(0.2, 0.2),      
                    scale=(0.7, 1.3),          
                    shear=(-20, 20)            
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
            ])
        else:
            # CIFAR10-specific augmentations
            return transforms.Compose([

                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),  
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
                
                # Add random erasing for occlusion robustness
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33))
            ])
   
    def train_epoch(self):
        """Train for one epoch"""
        self.encoder.train()
        self.projection_head.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            batch_size = data.size(0)
            

            views = [data] 
            
            # Add augmented views
            num_augmented_views = 3 
            for _ in range(num_augmented_views):
                views.append(torch.stack([self.augment(img) for img in data]))
            
            # Get embeddings for all views
            projections = []
            for view in views:
                z = self.encoder(view)
                p = self.projection_head(z)
                projections.append(p)
            
            # Compute loss between all pairs of views
            loss = 0
            num_pairs = 0
            for i in range(len(views)):
                for j in range(i+1, len(views)):
                    loss += self.criterion(projections[i], projections[j])
                    num_pairs += 1
            
            loss /= num_pairs
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Step the scheduler every batch update
            self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'[CONTRASTIVE] Epoch: {self.current_epoch} | '
                      f'Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Views: {len(views)}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model by computing embedding similarity on validation set"""
        self.encoder.eval()
        self.projection_head.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.val_loader):
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Create two augmented views
                augmented1 = torch.stack([self.augment(img) for img in data])
                augmented2 = torch.stack([self.augment(img) for img in data])
                
                # Get embeddings and projections
                h1 = self.encoder(augmented1)
                h2 = self.encoder(augmented2)
                z1 = self.projection_head(h1)
                z2 = self.projection_head(h2)
                
                # Compute loss
                loss = self.criterion(z1, z2)
                total_loss += loss.item()
        
        # Calculate validation loss
        val_loss = total_loss / len(self.val_loader)
        
        return val_loss, 0
    
    def train(self):
        """Main training loop"""
        print(f"[CONTRASTIVE] Starting contrastive training for {self.dataset_name}")
        print(f"[CONTRASTIVE] Training on {self.device} with batch size {self.batch_size} for {self.epochs} epochs")
        
        # Clear previous metrics
        self.train_losses = []
        self.val_losses = []
        
        # Track best model
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.epochs):
            # Update current epoch
            self.current_epoch = epoch + 1
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss, _ = self.validate()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print epoch summary
            print(f"[CONTRASTIVE] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    self.encoder,
                    self.model_save_path,
                    epoch=epoch,
                    loss=val_loss,
                    projection_head_state_dict=self.projection_head.state_dict()
                )
                print(f"[CONTRASTIVE] {self.dataset_name} - Saved model checkpoint with validation loss: {val_loss:.6f}")
        
        # Plot training curves
        self.plot_metrics(f'{self.dataset_name.lower()}_contrastive_curves.png')
        
        # Generate t-SNE visualization
        print(f"[CONTRASTIVE] Generating t-SNE visualization of latent space...")
        plot_tsne(self.encoder, self.test_loader, self.device, 
                 dataset_name=f"{self.dataset_name.lower()}_contrastive")
        
        return self.encoder 