import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_tsne, get_result_dir
from abc import ABC, abstractmethod
import time

class BaseTrainer(ABC):
    """Base class for all training procedures with refactored common logic"""
    
    def __init__(self, args, training_type):
        """
        Initialize the trainer with common parameters
        
        Args:
            args: Command line arguments
            training_type: Type of training (e.g., 'autoencoder', 'classifier', 'guided', 'contrastive')
        """
        self.args = args
        self.trainer_type = training_type.upper()
        self.device = args.device
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.result_dir = get_result_dir(args, training_type)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        if self.trainer_type in ['CLASSIFIER', 'GUIDED', 'CONTRASTIVE']: # Accuracy-based
             self.best_val_metric = float('-inf') # Higher is better (accuracy)
        else: # Loss-based (e.g., Autoencoder)
             self.best_val_metric = float('inf') # Lower is better (loss)
        self.best_val_acc = 0.0 # Keep tracking best accuracy regardless
        self.current_epoch = 0
        
        # These will be set by setup_dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset_name = None
        self.num_classes = None

        # These must be set by subclasses in _setup_models_optimizers
        self.model = None # Could be Autoencoder, Encoder, etc.
        self.encoder = None # Reference to encoder part if applicable
        self.classifier = None # Reference to classifier part if applicable
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.model_save_path = None # Subclass must define this

    def setup_dataloaders(self, setup):
        """Sets up dataloaders and dataset info from the setup dictionary"""
        self.train_loader = setup['train_loader']
        self.val_loader = setup['val_loader']
        self.test_loader = setup['test_loader']
        self.dataset_name = setup['dataset_name']
        self.num_classes = setup.get('num_classes', 10) # Default to 10 if not present
        print(f"[{self.trainer_type}] Dataloaders set up for {self.dataset_name}.")
        print(f"[{self.trainer_type}] Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}, Test batches: {len(self.test_loader)}")

    @abstractmethod
    def _setup_models_optimizers(self, setup):
        """
        Subclasses must implement this method to:
        1. Initialize self.model (and self.encoder/self.classifier if applicable)
        2. Initialize self.optimizer
        3. Initialize self.scheduler
        4. Initialize self.criterion
        5. Define self.model_save_path
        """
        pass

    @abstractmethod
    def _train_batch(self, batch):
        """
        Subclasses must implement this method to process a single training batch.
        Should perform forward pass, loss calculation, backward pass, optimizer step.
        Must return a dictionary containing at least 'loss', e.g., {'loss': batch_loss, 'accuracy': batch_acc}
        """
        pass

    @abstractmethod
    def _validate_epoch(self):
        """
        Subclasses must implement this method to perform a validation epoch.
        Should use self.evaluate() or custom logic.
        Must return a dictionary of validation metrics, e.g., {'loss': val_loss, 'accuracy': val_acc}
        """
        pass

    def train(self):
        """Main training loop"""
        print(f"\n[{self.trainer_type}] Starting training for {self.epochs} epochs on {self.device}...")
        start_time = time.time()

        # Determine the validation metric to monitor based on trainer type
        val_metric_name = 'accuracy' if self.trainer_type in ['classifier', 'guided', 'contrastive'] else 'loss'
        higher_is_better = val_metric_name == 'accuracy'

        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()

            # --- Training Epoch ---
            self.set_train_mode(True)
            total_train_loss = 0
            total_train_correct = 0
            total_train_samples = 0

            for batch_idx, batch in enumerate(self.train_loader):
                batch_metrics = self._train_batch(batch) # Delegate batch processing
                total_train_loss += batch_metrics['loss'] * self._get_batch_size(batch)
                if 'correct' in batch_metrics and 'total' in batch_metrics:
                    total_train_correct += batch_metrics['correct']
                    total_train_samples += batch_metrics['total']

                # Optional: Print batch progress (can be customized in _train_batch if needed)
                # if batch_idx % 100 == 0: print(...)

            avg_train_loss = total_train_loss / len(self.train_loader.dataset)
            avg_train_acc = (100. * total_train_correct / total_train_samples) if total_train_samples > 0 else 0.0
            self.train_losses.append(avg_train_loss)
            if total_train_samples > 0: self.train_accuracies.append(avg_train_acc)

            # --- Validation Epoch ---
            self.set_train_mode(False)
            val_metrics = self._validate_epoch() # Delegate validation
            val_loss = val_metrics.get('loss', float('inf'))
            val_acc = val_metrics.get('accuracy', 0.0)
            self.val_losses.append(val_loss)
            if 'accuracy' in val_metrics: self.val_accuracies.append(val_acc)

            # --- Scheduler Step ---
            if self.scheduler:
                # Handle different scheduler types
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[val_metric_name])
                else:
                    self.scheduler.step()

            # --- Logging ---
            epoch_time = time.time() - epoch_start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(f"[{self.trainer_type}] Epoch: {self.current_epoch}/{self.epochs} | "
                  f"Time: {epoch_time:.2f}s | LR: {lr:.6f} | "
                  f"Train Loss: {avg_train_loss:.4f}" + (f" | Train Acc: {avg_train_acc:.2f}%" if avg_train_acc > 0 else "") + " | "
                  f"Val Loss: {val_loss:.4f}" + (f" | Val Acc: {val_acc:.2f}%" if val_acc > 0 else ""))

            # --- Checkpoint Saving ---
            current_val_metric = val_metrics[val_metric_name]
            is_best = (current_val_metric > self.best_val_metric) if higher_is_better else (current_val_metric < self.best_val_metric)

            if is_best:
                self.best_val_metric = current_val_metric
                self.save_checkpoint(is_best=True)
                print(f"[{self.trainer_type}] Best model saved (Epoch {epoch+1}, Val {val_metric_name}: {self.best_val_metric:.4f})")
            # Optionally save latest checkpoint every N epochs or at the end
            # self.save_checkpoint(epoch, is_best=False)

        total_training_time = time.time() - start_time
        print(f"\n[{self.trainer_type}] Training finished in {total_training_time:.2f} seconds.")
        print(f"[{self.trainer_type}] Best Validation {val_metric_name}: {self.best_val_metric:.4f}")

        # --- Plotting ---
        self.plot_metrics()

        # --- Final Evaluation on Test Set ---
        print(f"\n[{self.trainer_type}] Evaluating on Test Set using best checkpoint...")
        self.load_checkpoint(load_best=True) # Load the best model
        test_metrics = self.evaluate(self.test_loader)
        print(f"[{self.trainer_type}] Test Results -> Loss: {test_metrics['loss']:.4f}" +
              (f" | Accuracy: {test_metrics['accuracy']:.2f}%" if 'accuracy' in test_metrics else ""))
        if 'class_accuracy' in test_metrics:
             for i, acc in enumerate(test_metrics['class_accuracy']):
                 print(f"  Class {i}: {acc:.2f}%")

        # Return the primary model component (e.g., encoder, classifier)
        return self._get_trained_model()

    def evaluate(self, data_loader):
        """Generic evaluation loop for validation or testing."""
        self.set_train_mode(False)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.no_grad():
            for batch in data_loader:
                batch_metrics = self._process_evaluation_batch(batch, class_correct, class_total)

                total_loss += batch_metrics['loss'] * batch_metrics['total']
                total_correct += batch_metrics['correct']
                total_samples += batch_metrics['total']

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = 100. * total_correct / total_samples if total_samples > 0 else 0
        class_accuracy = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                          for i in range(self.num_classes)]

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'correct': total_correct,
            'total': total_samples
        }

    def _process_evaluation_batch(self, batch, class_correct, class_total):
        """
        Processes a single batch during evaluation.
        Can be overridden by subclasses if specialized logic is needed,
        especially for how model outputs are obtained.
        """
        data, targets = self._unpack_batch(batch)
        data = data.to(self.device)
        targets = targets.to(self.device) if targets is not None else None

        # --- Get Model Outputs (Subclass might need to customize this part) ---
        # Default logic assumes a model that takes data and returns outputs
        # suitable for the criterion. Classifier/Guided models might need adjustment.
        if self.trainer_type == 'AUTOENCODER':
            outputs = self.model(data)
            loss = self.criterion(outputs, data) if self.criterion else torch.tensor(0.0)
            predicted = None # No classification accuracy for AE
            correct = 0
        elif self.trainer_type == 'CLASSIFIER':
            if self.encoder and self.encoder is not self.model:
                 self.encoder.eval()
            with torch.no_grad():
                 features = self.encoder(data)
                 features = F.normalize(features, p=2, dim=1)
            outputs = self.classifier(features)
            loss = self.criterion(outputs, targets) if self.criterion else torch.tensor(0.0)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
        elif self.trainer_type == 'GUIDED' or self.trainer_type == 'CONTRASTIVE': # Contrastive eval uses linear probe
             # For guided, use both trained encoder and classifier
             # For contrastive, assume a linear probe classifier has been attached for eval
            self.encoder.eval()
            with torch.no_grad():
                 features = self.encoder(data)
                 features = F.normalize(features, p=2, dim=1)

            if hasattr(self, 'classifier') and self.classifier:
                 self.classifier.eval()
                 outputs = self.classifier(features)
                 loss = self.criterion(outputs, targets) if self.criterion else torch.tensor(0.0)
                 _, predicted = outputs.max(1)
                 correct = predicted.eq(targets).sum().item()
            else:
                 outputs = features
                 loss = torch.tensor(0.0)
                 predicted = None
                 correct = 0
        else:
            # Fallback/Error
            raise NotImplementedError(f"Evaluation logic not defined for trainer type: {self.trainer_type}")

        batch_size = data.size(0)

        # Update per-class accuracy if applicable
        if predicted is not None and targets is not None:
            for c in range(self.num_classes):
                class_mask = (targets == c)
                class_total[c] += class_mask.sum().item()
                class_correct[c] += predicted[class_mask].eq(targets[class_mask]).sum().item()

        return {
            'loss': loss.item(),
            'correct': correct,
            'total': batch_size
        }

    def save_checkpoint(self, is_best=False):
        """Saves model checkpoint, optionally marking the best one."""
        if not self.model_save_path:
            print(f"[{self.trainer_type}] Warning: model_save_path not set. Cannot save checkpoint.")
            return

        # Ensure model_save_path includes dataset name and type for clarity
        base_filename = f"{self.dataset_name.lower()}_{self.trainer_type.lower()}"
        checkpoint_dir = os.path.join(self.result_dir, self.trainer_type.lower(), self.dataset_name.lower())
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define state to save (only state dicts and necessary info)
        state = {
            'epoch': self.current_epoch,
            'best_val_metric': self.best_val_metric,
            'best_val_acc': self.best_val_acc,
            # Save state dicts based on what exists
            'model_state_dict': self.model.state_dict() if self.model else None,
            'encoder_state_dict': self.encoder.state_dict() if self.encoder and self.encoder is not self.model else None,
            'classifier_state_dict': self.classifier.state_dict() if self.classifier else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            # Include trainer type for potential logic during loading
            'trainer_type': self.trainer_type
        }
        # Remove None values to keep checkpoint clean
        state = {k: v for k, v in state.items() if v is not None}


        latest_path = os.path.join(checkpoint_dir, f"{base_filename}_latest.pth")
        torch.save(state, latest_path)
        # print(f"[{self.trainer_type}] Checkpoint saved to {latest_path}")

        if is_best:
            best_path = os.path.join(checkpoint_dir, f"{base_filename}_best.pth")
            torch.save(state, best_path)
            # print(f"[{self.trainer_type}] Best model saved to {best_path}")

    def load_checkpoint(self, load_best=False):
        """Loads model checkpoint."""
        if not self.model_save_path:
            print(f"[{self.trainer_type}] Warning: model_save_path not set. Cannot load checkpoint.")
            return

        base_filename = f"{self.dataset_name.lower()}_{self.trainer_type.lower()}"
        checkpoint_dir = os.path.join(self.result_dir, self.trainer_type.lower(), self.dataset_name.lower())
        filename = f"{base_filename}_best.pth" if load_best else f"{base_filename}_latest.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        if os.path.isfile(checkpoint_path):
            try:
                # Load with weights_only=False for now if needed, but saved state is safer
                # For maximum safety, consider loading state dicts individually after checking keys
                print(f"[{self.trainer_type}] Loading checkpoint: {checkpoint_path}")
                # Use weights_only=None to use the default, or False if necessary for old checkpoints
                # Setting to False can execute arbitrary code, only do this if you trust the source!
                # Since we control the saving process now, True should work for new checkpoints.
                # Let's try True first for safety. If it fails for old checkpoints, consider False.
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True) # Try True first

                # Restore state carefully, checking if keys exist in the checkpoint
                self.current_epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
                self.best_val_metric = checkpoint.get('best_val_metric', self.best_val_metric)
                self.best_val_acc = checkpoint.get('best_val_acc', self.best_val_acc)

                # Load model states based on what's available in the checkpoint and trainer
                if 'model_state_dict' in checkpoint and self.model:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'encoder_state_dict' in checkpoint and self.encoder and self.encoder is not self.model:
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                if 'classifier_state_dict' in checkpoint and self.classifier:
                    self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

                if self.optimizer and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    # Be cautious loading scheduler state, especially if epoch/steps change
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except:
                        print(f"[{self.trainer_type}] Warning: Could not load scheduler state dict. May reset scheduler.")


                print(f"[{self.trainer_type}] Checkpoint loaded successfully (Epoch {self.current_epoch-1}). Resuming training.")

            except Exception as e:
                print(f"[{self.trainer_type}] Error loading checkpoint from {checkpoint_path}: {e}. Starting from scratch.")
                self.current_epoch = 0 # Reset epoch if loading failed
        else:
            print(f"[{self.trainer_type}] Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")

    def _load_pretrained_encoder(self, checkpoint_path):
        """Helper to load only encoder weights from a checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"[{self.trainer_type}] Warning: Pretrained encoder checkpoint not found at {checkpoint_path}.")
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                print(f"[{self.trainer_type}] Pretrained encoder loaded from {checkpoint_path}")
                return True
            elif 'model_state_dict' in checkpoint: # Try loading from a full model state dict if encoder specific isn't there
                 # This requires the state dict keys to match or needs careful mapping
                 # For simplicity, we assume 'encoder_state_dict' is the standard
                 print(f"[{self.trainer_type}] Warning: 'encoder_state_dict' not found in {checkpoint_path}. Trying 'model_state_dict'. May fail if keys don't match.")
                 # Attempt loading, might fail if keys mismatch (e.g. loading Autoencoder into Encoder)
                 try:
                     # Filter state dict for encoder keys if possible (depends on naming convention)
                     encoder_keys = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
                     if encoder_keys:
                          # Remove 'encoder.' prefix if necessary for the current self.encoder model
                          encoder_keys_renamed = {k.replace('encoder.', '', 1): v for k, v in encoder_keys.items()}
                          self.encoder.load_state_dict(encoder_keys_renamed)
                          print(f"[{self.trainer_type}] Pretrained encoder loaded from 'model_state_dict' in {checkpoint_path}")
                          return True
                     else:
                          print(f"[{self.trainer_type}] Could not find encoder keys in 'model_state_dict'.")
                          return False
                 except Exception as e:
                     print(f"[{self.trainer_type}] Failed to load encoder from 'model_state_dict': {e}")
                     return False

            else:
                print(f"[{self.trainer_type}] Warning: Could not find 'encoder_state_dict' or 'model_state_dict' in {checkpoint_path}.")
                return False
        except Exception as e:
            print(f"[{self.trainer_type}] Error loading pretrained encoder from {checkpoint_path}: {e}")
            return False

    def plot_metrics(self):
        """Plots training and validation loss and accuracy curves."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'{self.dataset_name} {self.trainer_type} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)


        if self.train_accuracies or self.val_accuracies: # Only plot accuracy if available
             plt.subplot(1, 2, 2)
             if self.train_accuracies: plt.plot(self.train_accuracies, label='Train Accuracy')
             if self.val_accuracies: plt.plot(self.val_accuracies, label='Validation Accuracy')
             plt.title(f'{self.dataset_name} {self.trainer_type} Accuracy')
             plt.xlabel('Epoch')
             plt.ylabel('Accuracy (%)')
             plt.legend()
             plt.grid(True)


        plt.tight_layout()
        plot_filename = f"{self.dataset_name.lower()}_{self.trainer_type.lower()}_curves.png"
        plt.savefig(os.path.join(self.result_dir, plot_filename))
        print(f"[{self.trainer_type}] Metrics plot saved to {os.path.join(self.result_dir, plot_filename)}")
        plt.close()

    def set_train_mode(self, train_mode=True):
        """Sets the model(s) to training or evaluation mode."""
        if self.model: self.model.train(train_mode)
        if self.encoder: self.encoder.train(train_mode)
        if self.classifier: self.classifier.train(train_mode)
        # Special handling for ClassifierTrainer where encoder is always eval
        if self.trainer_type == 'classifier' and self.encoder:
            self.encoder.eval()

    def _get_batch_size(self, batch):
        """Utility to get batch size from data tuple/list"""
        if isinstance(batch, (list, tuple)):
            # Assume first element is the primary data tensor
            return batch[0].size(0)
        elif isinstance(batch, torch.Tensor):
            return batch.size(0)
        return self.args.batch_size # Fallback

    def _unpack_batch(self, batch):
        """Unpacks batch into data and targets. Handles different batch formats."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1] # data, targets
            elif len(batch) == 1:
                return batch[0], None # Only data (e.g., Autoencoder)
            elif len(batch) == 3: # Potentially (view1, view2, targets) for contrastive
                 # Decide how contrastive batches are structured
                 # Option 1: Return (view1, view2), targets=None for _train_batch
                 # Option 2: Return combined data, targets for eval? Needs thought.
                 # For now, assume standard (data, targets) for eval compatibility
                 return batch[0], batch[2] # Return first view and targets for eval? Or handle in subclass?
            else:
                 raise ValueError(f"Unexpected batch format with {len(batch)} elements.")
        else: # Assume batch is just data tensor
            return batch, None

    def _get_trained_model(self):
        """Returns the primary trained model component(s)."""
        if self.trainer_type == 'autoencoder':
            return self.model # Return the full Autoencoder
        elif self.trainer_type == 'classifier':
            return self.classifier # Return the trained Classifier
        elif self.trainer_type == 'guided':
            return self.encoder, self.classifier # Return both
        elif self.trainer_type == 'contrastive':
            return self.encoder # Return the trained Encoder
        else:
            return self.model # Default fallback