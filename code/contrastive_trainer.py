import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from base_trainer import BaseTrainer
from losses import NTXentLoss # Import the new loss
from models import MNISTEncoder, CIFAR10Encoder, Classifier

class ContrastiveTrainer(BaseTrainer):
    """Trainer for contrastive self-supervised learning (SimCLR-style)"""

    def __init__(self, args, setup):
        super().__init__(args, 'contrastive')
        self.setup_dataloaders(setup) # Expects contrastive augmentations
        self._setup_models_optimizers(setup)

        # Optional: Setup linear probe classifier for validation accuracy
        self._setup_linear_probe()

        self.load_checkpoint() # Load checkpoint if exists

    def _setup_models_optimizers(self, setup):
        """Initializes Encoder, optimizer, scheduler, contrastive criterion."""
        encoder_class = MNISTEncoder if setup['is_mnist'] else CIFAR10Encoder
        # The 'model' from setup is the encoder
        self.encoder = setup['model'].to(self.device)
        self.model = self.encoder # BaseTrainer uses self.model sometimes

        # Contrastive Loss
        self.criterion = NTXentLoss(temperature=0.1, device=self.device) # Adjust temperature as needed

        # Optimizer for the encoder
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1e-3, weight_decay=1e-4) # Typical SimCLR settings
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.model_save_path = f"{self.dataset_name.lower()}_contrastive"
        print(f"[{self.trainer_type}] Encoder: {encoder_class.__name__} (Trainable), Criterion: NTXentLoss")

    def _setup_linear_probe(self):
        """Sets up a linear classifier for validation accuracy monitoring."""
        self.classifier = Classifier(self.args.latent_dim, self.num_classes).to(self.device)
        # We only train the *linear layer* of this classifier during validation
        # For simplicity here, we'll train the whole probe during validation epoch
        # A separate optimizer for the probe is cleaner
        self.probe_optimizer = optim.AdamW(self.classifier.parameters(), lr=1e-2, weight_decay=1e-5)
        self.probe_criterion = torch.nn.CrossEntropyLoss()
        print(f"[{self.trainer_type}] Linear probe classifier set up for validation.")


    def _unpack_batch(self, batch):
        """Handles contrastive batches: expects (view1, view2, targets)."""
        if len(batch) == 3:
            return batch[0], batch[1], batch[2] # view1, view2, targets
        else:
            # Fallback for standard evaluation loaders (data, targets)
            return batch[0], None, batch[1] # data, None, targets


    def _train_batch(self, batch):
        """Processes a single contrastive training batch."""
        view1, view2, _ = self._unpack_batch(batch) # Ignore targets for contrastive loss
        view1, view2 = view1.to(self.device), view2.to(self.device)

        # Forward pass: get features for both views
        self.optimizer.zero_grad()
        z_i = self.encoder(view1)
        z_j = self.encoder(view2)

        # Calculate contrastive loss
        loss = self.criterion(z_i, z_j)

        # Backward pass (updates encoder)
        loss.backward()
        self.optimizer.step()

        # Contrastive training doesn't typically track accuracy directly
        return {'loss': loss.item(), 'total': view1.size(0)}

    def _validate_epoch(self):
        """Performs validation: calculates loss and trains/evaluates linear probe."""
        self.set_train_mode(False) # Encoder to eval mode

        total_val_loss = 0
        total_probe_loss = 0
        total_probe_correct = 0
        total_probe_samples = 0

        # --- Optional: Train Linear Probe ---
        if hasattr(self, 'classifier') and self.classifier:
            self.classifier.train() # Set probe to train mode
            for batch in self.train_loader: # Use train loader for probe training
                data, _, targets = self._unpack_batch(batch) # Get single view + targets
                data, targets = data.to(self.device), targets.to(self.device)

                with torch.no_grad(): # Get features from frozen encoder
                    features = self.encoder(data)
                    features = F.normalize(features, p=2, dim=1)

                self.probe_optimizer.zero_grad()
                outputs = self.classifier(features.detach()) # Use detached features
                probe_loss = self.probe_criterion(outputs, targets)
                probe_loss.backward()
                self.probe_optimizer.step()
            self.classifier.eval() # Set probe back to eval mode

        # --- Evaluate Contrastive Loss & Probe Accuracy on Val Set ---
        with torch.no_grad():
            for batch in self.val_loader:
                view1, view2, targets = self._unpack_batch(batch)
                view1, view2 = view1.to(self.device), view2.to(self.device)
                targets = targets.to(self.device) if targets is not None else None

                # Contrastive validation loss
                z_i = self.encoder(view1)
                z_j = self.encoder(view2)
                val_loss = self.criterion(z_i, z_j)
                total_val_loss += val_loss.item() * view1.size(0)

                # Linear probe validation accuracy
                if hasattr(self, 'classifier') and self.classifier and targets is not None:
                    # Use view1 features for probe eval consistency
                    features = F.normalize(z_i, p=2, dim=1)
                    outputs = self.classifier(features)
                    probe_val_loss = self.probe_criterion(outputs, targets)
                    total_probe_loss += probe_val_loss.item() * view1.size(0)

                    _, predicted = outputs.max(1)
                    total_probe_correct += predicted.eq(targets).sum().item()
                    total_probe_samples += targets.size(0)


        avg_val_loss = total_val_loss / len(self.val_loader.dataset)
        avg_probe_acc = (100. * total_probe_correct / total_probe_samples) if total_probe_samples > 0 else 0.0
        avg_probe_loss = total_probe_loss / total_probe_samples if total_probe_samples > 0 else float('inf')

        print(f"[{self.trainer_type}] Validation -> Contrastive Loss: {avg_val_loss:.4f} | Probe Acc: {avg_probe_acc:.2f}%")

        # Return metrics for BaseTrainer (use probe accuracy as the primary metric)
        return {'loss': avg_val_loss, 'accuracy': avg_probe_acc, 'probe_loss': avg_probe_loss}

    # Override evaluate to use the linear probe if available
    def evaluate(self, data_loader):
        """Evaluate using the linear probe if available."""
        if hasattr(self, 'classifier') and self.classifier:
            print(f"[{self.trainer_type}] Evaluating using Linear Probe...")
            self.set_train_mode(False) # Encoder eval
            self.classifier.eval() # Classifier eval

            total_loss = 0
            total_correct = 0
            total_samples = 0
            class_correct = [0] * self.num_classes
            class_total = [0] * self.num_classes

            with torch.no_grad():
                for batch in data_loader:
                    data, _, targets = self._unpack_batch(batch) # Use single view
                    data, targets = data.to(self.device), targets.to(self.device)

                    features = self.encoder(data)
                    features = F.normalize(features, p=2, dim=1)
                    outputs = self.classifier(features)
                    loss = self.probe_criterion(outputs, targets)
                    _, predicted = outputs.max(1)

                    total_loss += loss.item() * data.size(0)
                    total_correct += predicted.eq(targets).sum().item()
                    total_samples += targets.size(0)

                    # Update per-class accuracy
                    for c in range(self.num_classes):
                         class_mask = (targets == c)
                         class_total[c] += class_mask.sum().item()
                         class_correct[c] += predicted[class_mask].eq(targets[class_mask]).sum().item()


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
        else:
            # Fallback if no probe exists (e.g., just evaluate contrastive loss)
            print(f"[{self.trainer_type}] No linear probe found, evaluating contrastive loss only.")
            # Implement contrastive loss calculation on the dataloader if needed
            return {'loss': float('inf'), 'accuracy': 0.0} # Placeholder

    def _get_trained_model(self):
        # Primary result of contrastive training is the encoder
        return self.encoder 