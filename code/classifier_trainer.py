import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from base_trainer import BaseTrainer
from models import MNISTEncoder, CIFAR10Encoder, Classifier
from utils import get_result_dir


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Applies mixup augmentation to a batch of features and labels."""
    # Sample lambda from beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    # Mix the features
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index] 
    return mixed_x, y_a, y_b, lam

class ClassifierTrainer(BaseTrainer):
    """Trainer for classifier on top of frozen encoder"""
    
    def __init__(self, args, setup):
        super().__init__(args, 'classifier')
        
        # Print CUDA status once during initialization
        print(f"[CLASSIFIER] Training on: {self.device}")
        
        # Get dataset info from setup
        self.dataset_name = setup['dataset_name']
        self.train_loader = setup['train_loader']
        self.val_loader = setup['val_loader']
        self.test_loader = setup['test_loader']
        self.num_classes = 10
        self.is_mnist = 'mnist' in self.dataset_name.lower()
        # Determine encoder type from args
        self.encoder_type = args.encoder_type       
        # Initialize the correct encoder architecture
        if self.is_mnist:
            self.encoder = MNISTEncoder(latent_dim=args.latent_dim).to(self.device)
        else:
            self.encoder = CIFAR10Encoder(latent_dim=args.latent_dim).to(self.device)
        
        # Create the classifier
        self.classifier = Classifier(args.latent_dim, self.num_classes).to(args.device)
        
        # Load pre-trained encoder weights
        self.load_pretrained_encoder(args)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        
        self.optimizer = optim.AdamW(
            self.classifier.parameters(), 
            lr=3e-3,  
            weight_decay=2e-3
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3, 
            steps_per_epoch=len(self.train_loader),
            epochs=self.epochs,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )
        
        self.model_save_path = f'{self.dataset_name.lower()}_classifier.pth'
        
        # Track per-class accuracies
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes
    
    def load_pretrained_encoder(self, args):
        """Load pre-trained encoder weights."""
        encoder_type = args.encoder_type
        dataset_name = self.dataset_name.lower()

        if encoder_type == 'self_supervised':
            # Path for self-supervised (autoencoder) model
            model_path = get_result_dir(args, 'self_supervised', True)
            model_path = os.path.join(model_path, f'{dataset_name}_autoencoder.pth')
            expected_key = 'encoder_state_dict' 
        elif encoder_type == 'contrastive':
            # Path for contrastive model
            model_path = get_result_dir(args, 'contrastive', True)
            model_path = os.path.join(model_path, f'{dataset_name}_contrastive.pth')
            expected_key = 'model_state_dict'
        else:
            raise ValueError(f"ERROR: Unknown encoder_type '{encoder_type}'")

        # Print the full path to help diagnose the issue
        print(f"Looking for pretrained {encoder_type} encoder at: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ERROR: No pretrained {encoder_type} encoder found at {model_path}. "
                                   f"Please train the {encoder_type} model first.")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # First try loading from expected key if checkpoint is a dict
            if isinstance(checkpoint, dict):
                if expected_key in checkpoint:
                    self.encoder.load_state_dict(checkpoint[expected_key])
                    print(f"Successfully loaded encoder from '{expected_key}'")
                    return
                elif 'encoder_state_dict' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    print("Successfully loaded encoder from 'encoder_state_dict'")
                    return
                # Try other common keys
                elif 'state_dict' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['state_dict'])
                    print("Successfully loaded encoder from 'state_dict'")
                    return
            
            # If checkpoint has encoder attribute (full model object)
            if hasattr(checkpoint, 'encoder'):
                self.encoder.load_state_dict(checkpoint.encoder.state_dict())
                print("Successfully loaded encoder from model object")
                return
            
            # Last resort: try loading checkpoint directly as state_dict
            try:
                self.encoder.load_state_dict(checkpoint)
                print("Successfully loaded encoder directly")
                return
            except Exception as e:
                if isinstance(checkpoint, dict):
                    raise ValueError(f"Could not load encoder. Available keys: {list(checkpoint.keys())}") from e
                else:
                    raise ValueError(f"Could not load encoder. Checkpoint type: {type(checkpoint)}") from e
                
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained {encoder_type} encoder: {str(e)}") from e
    
    def train_epoch(self):
        self.encoder.eval()
        self.classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Get total number of batches for percentage calculation
        total_batches = len(self.train_loader)
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass through encoder
            with torch.no_grad():
                features = self.encoder(data)
            self.optimizer.zero_grad()
            if np.random.random() > 0.5:
                # Get mixed features and mixed targets
                mixed_features, targets_a, targets_b, lam = mixup_data(
                    features, targets, alpha=0.2, device=self.device
                )
                
                outputs = self.classifier(mixed_features)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                outputs = self.classifier(features)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print batch progress (optional)
            if batch_idx % 100 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f"[CLASSIFIER] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                      f"Batch: {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, Current Train Acc: {current_acc:.2f}%, "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

        # Calculate epoch metrics AFTER the loop
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.classifier.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Reset per-class accuracy stats
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get features from encoder
                features = self.encoder(data)
                features = F.normalize(features, p=2, dim=1)
                
                # Forward pass
                outputs = self.classifier(features)
                loss = self.criterion(outputs, targets)
                
                # Compute metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                correct += batch_correct
                total += targets.size(0)
                
                # Update per-class accuracy stats
                for c in range(self.num_classes):
                    class_mask = (targets == c)
                    self.class_correct[c] += predicted[class_mask].eq(targets[class_mask]).sum().item()
                    self.class_total[c] += class_mask.sum().item()
                
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop"""
        print(f"[CLASSIFIER] Starting classifier training for {self.dataset_name}")
        print(f"[CLASSIFIER] Training on {self.device} with batch size {self.batch_size} for {self.epochs} epochs")
        print(f"[CLASSIFIER] Results directory: {self.result_dir}")
        
        # Initialize early stopping parameters
        best_val_accuracy = 0
        patience = 5 
        waiting = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Update current epoch
            self.current_epoch = epoch + 1
            
            # Train and validate
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Print epoch summary
            print(f"[CLASSIFIER] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_checkpoint(self.classifier, self.model_save_path, epoch=epoch, accuracy=val_accuracy)
                print(f"[CLASSIFIER] {self.dataset_name} - Saved model checkpoint with validation accuracy: {val_accuracy:.2f}%")
                waiting = 0  
            else:
                waiting += 1  
            
            # Early stopping check
            if waiting >= patience:
                print(f"[CLASSIFIER] Early stopping triggered - No improvement for {patience} epochs")
                break
        
        # Evaluate on test set
        self.evaluate_test()
        
        plot_path = f'{self.dataset_name.lower()}_classifier_curves.png'
        self.plot_metrics(plot_path)
        
        return self.classifier
    
    def evaluate_test(self):
        """Evaluate model on test set with detailed metrics"""
        self.encoder.eval()
        self.classifier.eval()
        
        # Prepare metric tracking
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        print(f"\n[CLASSIFIER] {self.dataset_name} - Evaluating on test set...")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get features from encoder
                features = self.encoder(data)
                
                # Forward pass through classifier
                outputs = self.classifier(features)
                loss = self.criterion(outputs, targets)
                
                # Compute metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                correct += batch_correct
                total += targets.size(0)
                
                # Per-class accuracy
                for c in range(self.num_classes):
                    class_mask = (targets == c)
                    class_correct[c] += (predicted[class_mask] == c).sum().item()
                    class_total[c] += class_mask.sum().item()
                
        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        # Print per-class accuracy
        print(f"[CLASSIFIER] {self.dataset_name} - Test per-class accuracy:")
        for i in range(self.num_classes):
            class_acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"    Class {i}: {class_acc:.2f}%")
        
        print(f"[CLASSIFIER] {self.dataset_name} - Test Loss: {avg_loss:.6f}, Test Acc: {accuracy:.2f}%")
        return accuracy
