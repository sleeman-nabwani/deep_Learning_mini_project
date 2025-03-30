import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_tsne, get_result_dir

class BaseTrainer:
    """Base class for all training procedures"""
    
    def __init__(self, args, training_type):
        """
        Initialize the trainer with common parameters
        
        Args:
            args: Command line arguments
            training_type: Type of training (self_supervised, classifier, classification_guided)
        """
        self.args = args
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
        self.best_val_metric = float('inf')
        self.best_val_acc = 0.0 
        self.current_epoch = 0
        
    def save_checkpoint(self, model, filename, **kwargs):
        """Save model checkpoint with additional data"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, os.path.join(self.result_dir, filename))
        
    def load_checkpoint(self, model, filename, map_location=None):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.result_dir, filename), 
                               map_location=map_location or self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Backward compatibility with older checkpoints
            model.load_state_dict(checkpoint)
            
        return checkpoint
    
    def extract_encoder_state_dict(self, checkpoint):
        """Extract encoder part from a full model state dict"""
        if 'encoder_state_dict' in checkpoint:
            return checkpoint['encoder_state_dict']
            
        encoder_state_dict = {}
        for key, value in checkpoint.items():
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format
                state_dict = checkpoint['model_state_dict']
            else:
                # Old format
                state_dict = checkpoint
                
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    encoder_key = key[len('encoder.'):]
                    encoder_state_dict[encoder_key] = value
                    
        return encoder_state_dict
    
    def plot_metrics(self, filename):
        """Plot training and validation metrics with enhanced visualization"""
        # Ensure result directory exists
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Debug information
        print(f"[DEBUG] Plotting metrics to {filename}")
        print(f"[DEBUG] Train losses: {len(self.train_losses)} points, values: {self.train_losses}")
        print(f"[DEBUG] Val losses: {len(self.val_losses)} points, values: {self.val_losses}")
        print(f"[DEBUG] Train accuracies: {len(self.train_accuracies)} points, values: {self.train_accuracies}")
        print(f"[DEBUG] Val accuracies: {len(self.val_accuracies)} points, values: {self.val_accuracies}")
        
        # Create figure with better settings
        plt.figure(figsize=(16, 8))
        plt.style.use('seaborn-v0_8-darkgrid')  # Use a better style
        
        # Plot losses with improved styling
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'o-', color='blue', linewidth=2, label='Training Loss')
        plt.plot(epochs, self.val_losses, 'o-', color='orange', linewidth=2, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Calculate appropriate y-axis limits for loss plot
        min_loss = min(min(self.train_losses), min(self.val_losses)) * 0.9
        max_loss = max(max(self.train_losses), max(self.val_losses)) * 1.1
        plt.ylim(min_loss, max_loss)
        
        # Add value annotations
        for i, (train_l, val_l) in enumerate(zip(self.train_losses, self.val_losses)):
            if i == 0 or i == len(self.train_losses) - 1:  # Only annotate first and last point
                plt.annotate(f'{train_l:.4f}', xy=(epochs[i], train_l), 
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
                plt.annotate(f'{val_l:.4f}', xy=(epochs[i], val_l), 
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Plot accuracies with improved styling if they exist
        if len(self.train_accuracies) > 0 and len(self.val_accuracies) > 0:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.train_accuracies, 'o-', color='green', linewidth=2, label='Training Accuracy')
            plt.plot(epochs, self.val_accuracies, 'o-', color='red', linewidth=2, label='Validation Accuracy')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # Calculate appropriate y-axis limits for accuracy plot
            min_acc = max(0, min(min(self.train_accuracies), min(self.val_accuracies)) * 0.9)
            max_acc = min(100, max(max(self.train_accuracies), max(self.val_accuracies)) * 1.1)
            plt.ylim(min_acc, max_acc)
            
            # Add value annotations
            for i, (train_a, val_a) in enumerate(zip(self.train_accuracies, self.val_accuracies)):
                if i == 0 or i == len(self.train_accuracies) - 1:  # Only annotate first and last point
                    plt.annotate(f'{train_a:.2f}%', xy=(epochs[i], train_a), 
                                xytext=(5, 5), textcoords='offset points', fontsize=9)
                    plt.annotate(f'{val_a:.2f}%', xy=(epochs[i], val_a), 
                                xytext=(5, 5), textcoords='offset points', fontsize=9)
        else:
            print("[WARNING] No accuracy data to plot")
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.result_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Enhanced plot saved to {save_path}")
        
        # Also display the plot if in an interactive environment
        try:
            plt.show()
        except:
            pass
        
        plt.close()
    
    def evaluate(self, model, data_loader, criterion=None, num_classes=10):
        """Evaluate model on given data loader"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Process batch and get metrics
                batch_loss, batch_correct, batch_total = self._process_evaluation_batch(
                    model, data, targets, criterion, class_correct, class_total, num_classes
                )
                
                total_loss += batch_loss
                correct += batch_correct
                total += batch_total
        
        # Calculate final metrics
        return self._calculate_evaluation_metrics(
            total_loss, correct, total, class_correct, class_total, 
            len(data_loader), num_classes
        )
    
    def _process_evaluation_batch(self, model, data, targets, criterion, class_correct, class_total, num_classes):
        """Process a single batch during evaluation"""
        outputs, loss, preds = self._get_outputs_and_predictions(model, data, targets, criterion)
        
        batch_loss = loss.item() if criterion else 0
        batch_correct = 0
        batch_total = 0
        
        # Calculate accuracy if we have predictions
        if preds is not None:
            batch_total = targets.size(0)
            batch_correct = preds.eq(targets).sum().item()
            
            # Update per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                if label < num_classes:  # Safety check
                    class_correct[label] += (preds[i] == targets[i]).item()
                    class_total[label] += 1
                    
        return batch_loss, batch_correct, batch_total
    
    def _get_outputs_and_predictions(self, model, data, targets, criterion):
        """Get model outputs, loss and predictions based on model type"""
        if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # Autoencoder
            outputs = model(data)
            loss = criterion(outputs, data) if criterion else 0
            preds = None
        elif hasattr(model, 'encoder') and hasattr(model, 'classifier'):
            # End-to-end classification model
            features = model.encoder(data)
            features = F.normalize(features, p=2, dim=1)
            outputs = model.classifier(features)
            loss = criterion(outputs, targets) if criterion else 0
            _, preds = outputs.max(1)
        else:
            # Simple classifier or encoder
            outputs = model(data)
            loss = criterion(outputs, targets) if criterion else 0
            _, preds = outputs.max(1) if outputs.size(1) > 1 else (outputs > 0.5).long()
            
        return outputs, loss, preds
    
    def _calculate_evaluation_metrics(self, total_loss, correct, total, class_correct, class_total, num_batches, num_classes):
        """Calculate final evaluation metrics"""
        avg_loss = total_loss / num_batches if total_loss > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        # Calculate per-class accuracy
        class_accuracy = [
            100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(num_classes)
        ]
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'correct': correct,
            'total': total
        }