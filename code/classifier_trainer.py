import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from base_trainer import BaseTrainer
from utils import plot_tsne

class ClassifierTrainer(BaseTrainer):
    """Trainer for classifier on top of frozen encoder"""
    
    def __init__(self, args, setup):
        super().__init__(args, 'classifier')
        self.encoder = setup['encoder'].to(args.device)
        self.dataset_name = setup['dataset_name']
        self.train_loader = setup['train_loader']
        self.val_loader = setup['val_loader']
        self.test_loader = setup['test_loader']
        self.num_classes = 10  
        
        from models import Classifier
        self.classifier = Classifier(args.latent_dim, self.num_classes).to(args.device)
        
        # Load pre-trained encoder if available
        self.load_pretrained_encoder(args)        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.optimizer = optim.AdamW(
            self.classifier.parameters(), 
            lr=3e-3,  
            weight_decay=5e-4  
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=5e-3,  
            steps_per_epoch=len(self.train_loader),
            epochs=self.epochs,
            pct_start=0.1,  
            div_factor=10,  
            final_div_factor=100  
        )
        
        # File naming
        self.model_save_path = f'{self.dataset_name.lower()}_classifier.pth'
    
    def load_pretrained_encoder(self, args):
        """Load pretrained encoder from autoencoder checkpoint"""
        # Corrected path - don't prepend result_dir
        autoencoder_path = os.path.join('results/self_supervised', 
                                     f'{self.dataset_name.lower()}', f'{self.dataset_name.lower()}_autoencoder.pth')
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(autoencoder_path, map_location=args.device)
            
            # Handle different checkpoint structures
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Extract just the encoder part
                encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                                  if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state_dict)
            else:
                # Old format or direct model state dict
                encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() 
                                  if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_state_dict)
            
            print(f"[CLASSIFIER] Successfully loaded pretrained encoder from {autoencoder_path}")
        except Exception as e:
            print(f"[CLASSIFIER] ERROR: Failed to load pretrained encoder: {e}")
            print("[CLASSIFIER] Training will continue with randomly initialized encoder")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Get total number of batches for percentage calculation
        total_batches = len(self.train_loader)
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass through encoder (with gradient disabled since it's frozen)
            with torch.no_grad():
                features = self.encoder(data)
                # Reshape if features is not already flat
                if len(features.shape) > 2:
                    features = features.reshape(features.size(0), -1)  # Flatten
                # Ensure features has the right size for the classifier
                if features.size(1) != self.args.latent_dim:
                    features = torch.nn.functional.adaptive_avg_pool1d(
                        features.unsqueeze(1), self.args.latent_dim).squeeze(1)
                features = F.normalize(features, p=2, dim=1)
            
            # Forward pass through classifier
            self.optimizer.zero_grad()
            outputs = self.classifier(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update LR scheduler every batch
            self.scheduler.step()
            
            # Compute metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(targets).sum().item()
            correct += batch_correct
            total += targets.size(0)
            batch_acc = 100. * batch_correct / targets.size(0)
            current_acc = 100. * correct / total
            
            # Print progress update every 100 batches or at specific percentages
            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                progress = 100. * batch_idx / total_batches
                print(f"[CLASSIFIER] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                      f"Batch: {batch_idx}/{total_batches} ({progress:.1f}%), "
                      f"Loss: {loss.item():.6f}, Batch Acc: {batch_acc:.2f}%, "
                      f"Current Train Acc: {current_acc:.2f}%, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Calculate average metrics
        avg_loss = total_loss / total_batches
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.classifier.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Encode data with the frozen encoder
                features = self.encoder(data)
                features = F.normalize(features, p=2, dim=1)
                
                # Forward pass
                outputs = self.classifier(features)
                loss = self.criterion(outputs, targets)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate per-class accuracy
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] += (predicted[i] == targets[i]).item()
                    class_total[label] += 1
        
        # Calculate validation metrics
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Print per-class accuracy
        print(f"[CLASSIFIER] {self.dataset_name} - Validation per-class accuracy:")
        for i in range(self.num_classes):
            class_acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"    Class {i}: {class_acc:.2f}%")
        
        return val_loss, val_acc
    
    def train(self):
        """Main training loop"""
        print(f"[CLASSIFIER] Starting training classifier for {self.dataset_name} with latent dim={self.args.latent_dim}")
        print(f"[CLASSIFIER] Training on {self.device} with batch size {self.batch_size} for {self.epochs} epochs")
        
        # Clear previous metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Initialize current epoch properly
        self.current_epoch = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Update current epoch BEFORE training
            self.current_epoch = epoch + 1  # Start from 1 instead of 0
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Store metrics - CRITICAL for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"[CLASSIFIER] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model (based on validation accuracy)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(self.classifier, self.model_save_path, epoch=epoch, accuracy=val_acc, loss=val_loss)
                print(f"[CLASSIFIER] {self.dataset_name} - Saved model checkpoint with validation accuracy: {val_acc:.2f}%")
        
        # Evaluate on test set
        self.evaluate_test()
        
        # Plot training curves with enhanced visualization
        self.plot_metrics(f'{self.dataset_name.lower()}_classifier_curves.png')
        
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
                
                # Get features from encoder and preprocess
                features = self.encoder(data)
                # Reshape if features is not already flat
                if len(features.shape) > 2:
                    features = features.reshape(features.size(0), -1)  # Flatten
                # Ensure features has the right size for the classifier
                if features.size(1) != self.args.latent_dim:
                    features = torch.nn.functional.adaptive_avg_pool1d(
                        features.unsqueeze(1), self.args.latent_dim).squeeze(1)
                features = F.normalize(features, p=2, dim=1)
                
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