import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from base_trainer import BaseTrainer
from utils import plot_tsne

class AutoencoderTrainer(BaseTrainer):
    """Trainer for self-supervised autoencoder"""
    
    def __init__(self, args, setup):
        super().__init__(args, 'self_supervised')
        
        # Print CUDA status once during initialization
        print(f"[AUTOENCODER] Training on: {self.device}")
        
        self.model = setup['model']
        # Explicitly ensure model is on the correct device
        self.model = self.model.to(self.device)
        
        self.dataset_name = setup['dataset_name']
        self.img_size = setup['img_size']
        self.train_loader = setup['train_loader']
        self.val_loader = setup['val_loader'] 
        self.test_loader = setup['test_loader']
        self.is_mnist = setup['is_mnist']
        
        # Basic training components
        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-3,  
            weight_decay=1e-4,  
            betas=(0.9, 0.95)  
        )
        
        # For advanced reconstruction quality, a different scheduler can help:
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  
            T_mult=2,  
            eta_min=1e-6  
        )
        
        # File naming
        self.model_save_path = f'{self.dataset_name.lower()}_autoencoder.pth'
    
    def validate_model(self):
        """Validate model architecture"""
        model, device = self.model, self.device
        
        # Get a batch from the train loader
        for data, _ in self.train_loader:
            data = data.to(device)
            
            # Ensure the model produces outputs of the correct shape
            with torch.no_grad():
                output = model(data)
                
            if output.shape != data.shape:
                raise ValueError(f"Model output shape {output.shape} doesn't match input shape {data.shape}")
                
            print(f"[AUTOENCODER] Model validation successful. "
                  f"Input shape: {data.shape}, Output shape: {output.shape}")
            break
    
    def train_epoch(self):
        """Train for one epoch with simplified tracking"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Get total number of batches
        total_batches = len(self.train_loader)
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Ensure correct output dimensions
            if outputs.shape != data.shape:
                outputs = F.interpolate(outputs, size=data.shape[2:], mode='bilinear', align_corners=False)
                
            loss = self.criterion(outputs, data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Simple stats tracking
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            
            # Print progress
            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                progress = 100. * batch_idx / total_batches
                print(f"[AUTOENCODER] Epoch {self.current_epoch}/{self.epochs} - "
                     f"Batch {batch_idx}/{total_batches} ({progress:.1f}%) - "
                     f"Loss: {loss_val:.6f}, "
                     f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        model, device = self.model, self.device
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(device)
                
                # Forward pass
                outputs = model(data)
                
                # Ensure correct output dimensions
                if outputs.shape != data.shape:
                    outputs = F.interpolate(outputs, size=data.shape[2:], mode='bilinear', align_corners=False)
                
                loss = self.criterion(outputs, data)
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate validation metrics
        val_loss = total_loss / num_batches
        
        return val_loss
    
    def plot_reconstructions(self, num_images=5):
        """Plot original and reconstructed images"""
        model, device = self.model, self.device
        model.eval()
        
        # Get batch from test loader
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        
        with torch.no_grad():
            # Select subset of images
            images = images[:num_images].to(device)
            batch_labels = labels[:num_images]
            
            # Get reconstructions
            reconstructions = model(images)
            
            # Ensure correct dimensions
            if reconstructions.shape != images.shape:
                reconstructions = F.interpolate(reconstructions, size=images.shape[2:], 
                                               mode='bilinear', align_corners=False)
            
            # Convert to numpy for plotting
            images_np = images.cpu().numpy()
            reconstructions_np = reconstructions.cpu().numpy()

            # Adjust shape for plotting
            if self.is_mnist:
                # MNIST has 1 channel
                images_np = images_np.reshape(-1, 28, 28)
                reconstructions_np = reconstructions_np.reshape(-1, 28, 28)
            else:
                # CIFAR10 has 3 channels
                images_np = np.transpose(images_np, (0, 2, 3, 1))
                reconstructions_np = np.transpose(reconstructions_np, (0, 2, 3, 1))
                
                # De-normalize if needed
                if images_np.min() < 0:
                    images_np = (images_np + 1) / 2
                    reconstructions_np = (reconstructions_np + 1) / 2
                
                # Clip to valid range
                images_np = np.clip(images_np, 0, 1)
                reconstructions_np = np.clip(reconstructions_np, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
        
        # Plot original images
        for i in range(num_images):
            if self.is_mnist:
                axes[0, i].imshow(images_np[i], cmap='gray')
                axes[1, i].imshow(reconstructions_np[i], cmap='gray')
            else:
                axes[0, i].imshow(images_np[i])
                axes[1, i].imshow(reconstructions_np[i])
            
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            
        axes[0, 0].set_ylabel('Original')
        axes[1, 0].set_ylabel('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, f'{self.dataset_name.lower()}_reconstructions.png'))
        plt.close()
        
        if self.is_mnist:
            idx1, idx2 = 0, 1
            
            for i in range(num_images):
                for j in range(i+1, num_images):
                    if batch_labels[i].item() != batch_labels[j].item():
                        idx1, idx2 = i, j
                        break
                if idx1 != 0 or idx2 != 1:
                    break
                
            img1, img2 = images[idx1], images[idx2]
            label1, label2 = batch_labels[idx1].item(), batch_labels[idx2].item()
            
            interpolated_outputs = []
            alphas = torch.linspace(0, 1, 10)
            
            with torch.no_grad():
                z1 = model.encoder(img1.unsqueeze(0))
                z2 = model.encoder(img2.unsqueeze(0))
                
                for alpha in alphas:
                    z = (1 - alpha) * z1 + alpha * z2
                    x_recon = model.decoder(z)
                    interpolated_outputs.append(x_recon.squeeze(0))
            
            save_dir = os.path.join(self.result_dir, "interpolations")
            os.makedirs(save_dir, exist_ok=True)
            
            fig, axes = plt.subplots(1, len(interpolated_outputs), figsize=(2 * len(interpolated_outputs), 2))
            fig.suptitle(f"MNIST Linear Interpolation: Label {label1} → {label2}", fontsize=14)

            for i, recon_img in enumerate(interpolated_outputs):
                recon_img_np = recon_img.cpu().numpy()
                recon_img_np = recon_img_np.reshape(28, 28)
                axes[i].imshow(recon_img_np, cmap="gray")
                axes[i].axis("off")
                axes[i].set_title(f"α={alphas[i]:.2f}")

            plt.tight_layout()
            save_file = os.path.join(save_dir, f"mnist_linear_interp_from_vis_{label1}_to_{label2}.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            
            print(f"[AUTOENCODER] Created MNIST interpolation from label {label1} to {label2}")
            print(f"[AUTOENCODER] Saved to {save_file}")
    
    def train(self):
        """Main training loop"""
        print(f"[AUTOENCODER] Starting autoencoder training for {self.dataset_name}")
        print(f"[AUTOENCODER] Training on {self.device} with batch size {self.batch_size} for {self.epochs} epochs")
        print(f"[AUTOENCODER] Saving results to: {self.result_dir}")
        
        # Validate that the model produces correct output shapes
        self.validate_model()
        
        # Store metrics for plotting
        self.train_losses = []
        self.val_losses = []
        
        # Set up best model tracking
        best_val_loss = float('inf')
        
        # Initialize current_epoch counter
        self.current_epoch = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Update current epoch
            self.current_epoch = epoch + 1
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Step the scheduler once per epoch
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print epoch summary
            print(f"[AUTOENCODER] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save more comprehensive checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'decoder_state_dict': self.model.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,

                }, os.path.join(self.result_dir, self.model_save_path))
                print(f"[AUTOENCODER] Best model saved with validation loss: {best_val_loss:.6f}")
            
            # Plot reconstructions at the end of training
            if self.current_epoch == self.epochs:
                self.plot_reconstructions()
                
        # Load best model for return
        best_checkpoint = torch.load(os.path.join(self.result_dir, self.model_save_path))
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Plot training curves
        self.plot_metrics(f'{self.dataset_name.lower()}_autoencoder_curves.png')
        
        # Generate t-SNE visualization
        print("[AUTOENCODER] Generating t-SNE visualization of latent space...")
        # For autoencoder, we need to use just the encoder part for t-SNE
        plot_tsne(self.model.encoder, self.test_loader, self.device,
                 dataset_name=f"{self.dataset_name.lower()}_autoencoder")
        
        return self.model