import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from base_trainer import BaseTrainer
from utils import plot_tsne, plot_reconstructions

# Assuming Autoencoder model is defined in models.py
from models import MNISTAutoencoder, CIFAR10Autoencoder

class AutoencoderTrainer(BaseTrainer):
    """Trainer for self-supervised autoencoder using the refactored BaseTrainer"""

    def __init__(self, args, setup):
        super().__init__(args, 'autoencoder')
        self.setup_dataloaders(setup) 
        self.mean = torch.tensor(setup['mean']).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor(setup['std']).view(-1, 1, 1).to(self.device)
        self._setup_models_optimizers(setup)

        # Load checkpoint if exists
        self.load_checkpoint()

    def _setup_models_optimizers(self, setup):
        """Initializes Autoencoder model, optimizer, scheduler, and criterion."""
        # Determine model class based on dataset
        model_class = MNISTAutoencoder if setup['is_mnist'] else CIFAR10Autoencoder
        self.model = model_class(self.args.latent_dim).to(self.device)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.model_save_path = f"{self.dataset_name.lower()}_autoencoder" # Base name for checkpoints
        print(f"[{self.trainer_type}] Model: {model_class.__name__}, Criterion: L1Loss, Optimizer: AdamW, Scheduler: CosineAnnealingLR")


    def _train_batch(self, batch):
        """Processes a single training batch for the Autoencoder."""
        data, _ = self._unpack_batch(batch) # Autoencoder only needs data
        data = data.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        reconstructions = self.model(data)
        loss = self.criterion(reconstructions, data)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'total': data.size(0)} # Return loss and batch size

    def _validate_epoch(self):
        """Performs a validation epoch using the generic evaluate method."""
        # The generic evaluate method in BaseTrainer handles AE if trainer_type is AUTOENCODER
        val_metrics = self.evaluate(self.val_loader)
        # Optionally plot reconstructions during validation
        if self.current_epoch % 5 == 0 or self.current_epoch == self.epochs: # Plot every 5 epochs
             self.plot_reconstructions(num_images=10)
        return val_metrics # evaluate returns a dict {'loss': ..., 'accuracy': ...}

    # Keep specific methods like plot_reconstructions if needed
    def plot_reconstructions(self, num_images=10):
        """Plots original and reconstructed images after denormalizing."""
        self.set_train_mode(False)
        # Use a fixed batch from validation loader for consistent visualization
        if not hasattr(self, '_fixed_plot_batch'):
             # Ensure the fixed batch is moved to the correct device initially
             self._fixed_plot_batch = next(iter(self.val_loader))
             # Move data and potentially targets to the trainer's device
             data_fixed, targets_fixed = self._unpack_batch(self._fixed_plot_batch)
             self._fixed_plot_batch = (data_fixed.to(self.device), targets_fixed.to(self.device) if targets_fixed is not None else None)


        data, _ = self._fixed_plot_batch # Get fixed batch (already on device)
        data = data[:num_images] # Select number of images

        with torch.no_grad():
            reconstructions = self.model(data)

        # Denormalize images for plotting
        # Ensure mean/std are on the same device as data
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        originals_denorm = data * std + mean
        reconstructions_denorm = reconstructions * std + mean

        # Clip to [0, 1] AFTER denormalization
        originals_denorm = torch.clamp(originals_denorm, 0, 1)
        reconstructions_denorm = torch.clamp(reconstructions_denorm, 0, 1)

        # --- Plotting Logic ---
        # Prepare save path (using the corrected path logic)
        plot_dir = os.path.join(self.result_dir, self.trainer_type.lower(), self.dataset_name.lower())
        os.makedirs(plot_dir, exist_ok=True)
        # Consistent naming for plot file
        save_path = os.path.join(plot_dir, f"{self.dataset_name.lower()}_{self.trainer_type.lower()}_reconstructions_epoch{self.current_epoch}.png")

        # Call the actual plotting function with DENORMALIZED tensors
        try:
            from utils import plot_reconstructions_impl
            plot_reconstructions_impl(
                originals_denorm.cpu(),      # Pass denormalized originals
                reconstructions_denorm.cpu(),# Pass denormalized reconstructions
                self.current_epoch,
                save_path,
                num_images=num_images        # Pass num_images to plotting function
            )
            print(f"[{self.trainer_type}] Plotted reconstructions for epoch {self.current_epoch} to {save_path}")
        except ImportError:
             print(f"[{self.trainer_type}] Warning: plot_reconstructions_impl not found in utils. Skipping plotting.")
        except Exception as plot_e:
             print(f"[{self.trainer_type}] Error during plotting: {plot_e}")


    # Optional: Override _get_trained_model if you want to return encoder/decoder separately
    # def _get_trained_model(self):
