import torch
import numpy as np
import random
import argparse
import os
from autoencoder_trainer import AutoencoderTrainer
from classifier_trainer import ClassifierTrainer
from classification_guided_trainer import ClassificationGuidedTrainer
from contrastive_trainer import ContrastiveTrainer
from utils import setup_datasets

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    """Fix all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    
    # Training type options - only select one
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Train autoencoder with reconstruction objective and then classifier')
    parser.add_argument('--contrastive', action='store_true', default=False,
                        help='Train encoder using contrastive learning objective and then classifier')
    parser.add_argument('--classification-guided', action='store_true', default=False,
                        help='Train encoder jointly with classifier for classification objective')
    
    # Standalone classifier training
    parser.add_argument('--train-classifier', action='store_true', default=False,
                        help='Train a classifier on top of a previously trained encoder')
    parser.add_argument('--encoder-type', type=str, choices=['self-supervised', 'contrastive'],
                        default='self-supervised', help='Type of encoder to use for classifier training')
    
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Select appropriate transform based on dataset
    dataset_name = "MNIST" if args.mnist else "CIFAR10"
    
    print(f"=== {dataset_name} TRAINING ===")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Data path: {args.data_path}")
    
    # Normalize encoder type
    if args.encoder_type == 'self-supervised':
        args.encoder_type = 'self_supervised'
    args.is_classifier = False
    
    # Train the autoencoder in self-supervised mode
    if args.self_supervised:
        print("\n=== STARTING AUTOENCODER TRAINING ===")
        setup = setup_datasets(args, model_type='autoencoder')
        trainer = AutoencoderTrainer(args, setup)
        autoencoder = trainer.train()
        print("=== AUTOENCODER TRAINING COMPLETE ===\n")
        
        # Always train classifier on top of self-supervised encoder
        print("\n=== STARTING CLASSIFIER TRAINING ON AUTOENCODER ENCODER ===")
        setup = setup_datasets(args, model_type='encoder')
        setup['encoder'] = autoencoder.encoder  # Use the trained encoder
        setup['encoder_type'] = 'self_supervised'
        args.encoder_type = 'self_supervised'
        args.is_classifier = True
        trainer = ClassifierTrainer(args, setup)
        classifier = trainer.train()
        print("=== CLASSIFIER TRAINING COMPLETE ===\n")
    
    # Train using contrastive learning approach
    elif args.contrastive:
        print("\n=== STARTING CONTRASTIVE LEARNING TRAINING ===")
        setup = setup_datasets(args, model_type='contrastive')
        trainer = ContrastiveTrainer(args, setup)
        encoder = trainer.train()
        print("=== CONTRASTIVE LEARNING TRAINING COMPLETE ===\n")
        
        # Always train classifier on top of contrastive encoder
        print("\n=== STARTING CLASSIFIER TRAINING ON CONTRASTIVE ENCODER ===")
        setup = setup_datasets(args, model_type='encoder')
        setup['encoder'] = encoder
        setup['encoder_type'] = 'contrastive'
        args.encoder_type = 'contrastive'
        args.is_classifier = True
        trainer = ClassifierTrainer(args, setup)
        classifier = trainer.train()
        print("=== CLASSIFIER TRAINING COMPLETE ===\n")
    
    # Train with classification guidance (joint training)
    elif args.classification_guided:
        print("\n=== STARTING CLASSIFICATION-GUIDED TRAINING ===")
        setup = setup_datasets(args, model_type='classification_guided')
        trainer = ClassificationGuidedTrainer(args, setup)
        encoder, classifier = trainer.train()
        print("=== CLASSIFICATION-GUIDED TRAINING COMPLETE ===\n")
    
    # Train only a classifier on a specified encoder type (using previously trained model)
    elif args.train_classifier:
        encoder_type = args.encoder_type
        args.is_classifier = True
        print(f"\n=== STARTING CLASSIFIER TRAINING ON PREVIOUSLY TRAINED {encoder_type.upper()} ENCODER ===")
        
        setup = setup_datasets(args, model_type='encoder')
        setup['encoder_type'] = encoder_type
        
        trainer = ClassifierTrainer(args, setup)
        classifier = trainer.train()
        print("=== CLASSIFIER TRAINING COMPLETE ===\n")
    
    else:
        print("No training mode specified. Please use --self-supervised, --contrastive, --classification-guided, or --train-classifier")

