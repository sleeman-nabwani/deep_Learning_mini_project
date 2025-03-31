import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from base_trainer import BaseTrainer
from utils import plot_tsne
from models import Classifier, MNISTEncoder, CIFAR10Encoder

class ClassifierTrainer(BaseTrainer):
    """Trainer for classifier on top of frozen encoder using refactored BaseTrainer"""
    
    def __init__(self, args, setup):
        super().__init__(args, 'classifier')
        self.setup_dataloaders(setup)
        self._setup_models_optimizers(setup)

        # Load pretrained encoder *before* potentially loading classifier checkpoint
        self._load_pretrained_encoder_from_args()

        # Freeze encoder (BaseTrainer's set_train_mode handles this based on type)
        self.set_train_mode(False) # Initial mode setup

        # Load classifier checkpoint if exists (will load optimizer/scheduler too)
        self.load_checkpoint()

    def _setup_models_optimizers(self, setup):
        """Initializes Encoder, Classifier, optimizer, scheduler, criterion."""
        # Setup Encoder (will be frozen)
        encoder_class = MNISTEncoder if setup['is_mnist'] else CIFAR10Encoder
        # We need the encoder instance, but BaseTrainer expects self.model or self.encoder/self.classifier
        # Let's store it in self.encoder directly. BaseTrainer's evaluate uses it.
        self.encoder = encoder_class(self.args.latent_dim).to(self.device)

        # Setup Classifier (trainable)
        self.classifier = Classifier(self.args.latent_dim, self.num_classes).to(self.device)

        # Define criterion, optimizer, scheduler (only for classifier parameters)
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

        self.model_save_path = f"{self.dataset_name.lower()}_classifier"
        print(f"[{self.trainer_type}] Encoder: {encoder_class.__name__} (Frozen), Classifier: Classifier, Criterion: CrossEntropyLoss, Optimizer: AdamW, Scheduler: OneCycleLR")

    def _load_pretrained_encoder_from_args(self):
        """Loads the pretrained encoder specified by convention or args."""
        # Determine expected pretrained encoder path (e.g., from autoencoder)
        # Convention: Use the best autoencoder checkpoint
        encoder_model_name = f"{self.dataset_name.lower()}_autoencoder_best.pth"
        encoder_path = os.path.join(self.result_dir, encoder_model_name)
        print(f"[{self.trainer_type}] Attempting to load pretrained encoder from: {encoder_path}")
        loaded = self._load_pretrained_encoder(encoder_path)
        if not loaded:
             print(f"[{self.trainer_type}] WARNING: Pretrained encoder not loaded. Classifier training will use randomly initialized encoder features (which is unusual).")
             # Decide if this should be an error or just a warning
             # raise FileNotFoundError(f"Pretrained encoder checkpoint not found at {encoder_path}")

    def _train_batch(self, batch):
        """Processes a single training batch for the Classifier (encoder frozen)."""
        data, targets = self._unpack_batch(batch)
        data, targets = data.to(self.device), targets.to(self.device)

        # --- Forward pass ---
        self.optimizer.zero_grad()

        # Encoder features (encoder is frozen via set_train_mode)
        with torch.no_grad(): # Explicit no_grad for encoder forward pass
             features = self.encoder(data)
             features = F.normalize(features, p=2, dim=1) # Normalize features

        # Classifier forward pass (trainable)
        outputs = self.classifier(features)
        loss = self.criterion(outputs, targets)

        # --- Backward pass (only affects classifier) ---
        loss.backward()
        self.optimizer.step()

        # --- Scheduler Step (if per-batch like OneCycleLR) ---
        if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
             self.scheduler.step()

        # --- Metrics ---
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = 100. * correct / total

        return {'loss': loss.item(), 'correct': correct, 'total': total, 'accuracy': accuracy}

    def _validate_epoch(self):
        """Performs a validation epoch using the generic evaluate method."""
        # BaseTrainer's evaluate method handles CLASSIFIER type correctly
        val_metrics = self.evaluate(self.val_loader)
        return val_metrics

    # evaluate_test is now handled by the base class calling evaluate(self.test_loader)
    # plot_tsne can be called from main script or after training if needed