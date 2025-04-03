import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from base_trainer import BaseTrainer
from utils import plot_tsne
import numpy as np

class ClassificationGuidedTrainer(BaseTrainer):
    """Trainer for joint encoder and classifier (classification-guided training)"""
    
    def __init__(self, args, setup):
        super().__init__(args, 'classification_guided')
        
        print(f"[GUIDED] Training on: {self.device}")
        
        self.encoder = setup['model']  

        self.encoder = self.encoder.to(self.device)
        
        self.dataset_name = setup['dataset_name']
        self.train_loader = setup['train_loader']
        self.val_loader = setup['val_loader']
        self.test_loader = setup['test_loader']
        self.num_classes = setup.get('num_classes', 10)
        
        # Create classifier
        from models import Classifier
        self.classifier = Classifier(args.latent_dim, self.num_classes).to(self.device)
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        self.optimizer = optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=0.001, weight_decay=1e-4)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6)
        
        # File naming
        self.model_save_path = f'{self.dataset_name.lower()}_guided.pth'
    
    def train_epoch(self):
        """Train for one epoch"""
        self.encoder.train()
        self.classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            features = self.encoder(data)
            features = F.normalize(features, p=2, dim=1)
            outputs = self.classifier(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print batch progress
            if batch_idx % 100 == 0:
                current_acc = 100. * correct / total if total > 0 else 0
                print(f"[GUIDED] {self.dataset_name} - Epoch: {len(self.train_losses) + 1}/{self.epochs}, "
                      f"Batch: {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, Current Train Acc: {current_acc:.2f}%, "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.encoder.eval()
        self.classifier.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                features = self.encoder(data)
                features = F.normalize(features, p=2, dim=1)
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
        
        return val_loss, val_acc
    
    def train(self):
        """Main training loop"""
        print(f"[GUIDED] Starting classification-guided training for {self.dataset_name}")
        print(f"[GUIDED] Training on {self.device} with batch size {self.batch_size} for {self.epochs} epochs")
        
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
            
            # Step the scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"[GUIDED] {self.dataset_name} - Epoch: {self.current_epoch}/{self.epochs}, "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model (based on validation accuracy)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                
                # Save model in standardized format
                self.save_checkpoint(
                    self.encoder, 
                    self.model_save_path,
                    epoch=epoch,
                    accuracy=val_acc,
                    loss=val_loss,
                    classifier_state_dict=self.classifier.state_dict()
                )
                print(f"[GUIDED] {self.dataset_name} - Saved model checkpoint with validation accuracy: {val_acc:.2f}%")
        
        # Load best model for evaluation
        checkpoint = self.load_checkpoint(self.encoder, self.model_save_path)
        if 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Evaluate on test set
        self.test_accuracy = self.evaluate_test()
        
        # Plot training curves
        self.plot_metrics(f'{self.dataset_name.lower()}_guided_curves.png')
        
        # Generate t-SNE visualization
        plot_tsne(self.encoder, self.test_loader, self.device, 
                 dataset_name=f"{self.dataset_name.lower()}_guided")
        
        return self.encoder, self.classifier
    
    def evaluate_test(self):
        """Evaluate on test dataset"""
        self.encoder.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                features = self.encoder(data)
                features = F.normalize(features, p=2, dim=1)
                outputs = self.classifier(features)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate per-class accuracy
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] += (predicted[i] == targets[i]).item()
                    class_total[label] += 1
        
        test_acc = 100. * correct / total
        print(f"[GUIDED] {self.dataset_name} - Testing completed!")
        print(f"[GUIDED] {self.dataset_name} - Test Accuracy: {test_acc:.2f}%")
        print(f"[GUIDED] {self.dataset_name} - Per-class test accuracy:")
        for i in range(self.num_classes):
            class_acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"    Class {i}: {class_acc:.2f}%")
        return test_acc