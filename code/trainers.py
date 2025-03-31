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
        self.best_val_metric = float('inf')  # For loss minimization
        self.best_val_acc = 0.0  # For accuracy maximization
        
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
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracies if they exist
        if self.train_accuracies and self.val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(self.train_accuracies, label='Training Accuracy')
            plt.plot(self.val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, filename))
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