import torch
import argparse
import random
import numpy as np
import os

# Import trainers and setup function
from utils import setup_datasets
from autoencoder_trainer import AutoencoderTrainer
from classifier_trainer import ClassifierTrainer
from classification_guided_trainer import ClassificationGuidedTrainer
from contrastive_trainer import ContrastiveTrainer # Import new trainer

def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Learning Framework")

    # --- Core Configuration ---
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="./datasets", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False, help='Use MNIST dataset (default: CIFAR10)')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')

    # --- Training Modes ---
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Train self-supervised autoencoder (reconstruction objective)')
    parser.add_argument('--contrastive', action='store_true', default=False,
                        help='Train self-supervised encoder (contrastive objective - SimCLR style)')
    parser.add_argument('--classification-guided', action='store_true', default=False,
                        help='Train encoder jointly with classifier (classification objective)')

    # --- Secondary Training Mode (Requires a Pretrained Encoder) ---
    parser.add_argument('--train-classifier', action='store_true', default=False,
                        help='Train a classifier on top of a *pretrained* encoder (requires a prior run or saved checkpoint)')

    # --- Argument Validation ---
    args = parser.parse_args()
    primary_modes = [args.self_supervised, args.contrastive, args.classification_guided]
    if sum(primary_modes) > 1:
        print("Warning: Multiple primary training modes selected (--self-supervised, --contrastive, --classification-guided). Prioritizing...")
        if args.contrastive:
            args.self_supervised = False
            args.classification_guided = False
            print("Prioritizing --contrastive training.")
        elif args.classification_guided:
             args.self_supervised = False
             print("Prioritizing --classification-guided training.")

    if args.train_classifier and not (args.self_supervised or args.contrastive or args.classification_guided):
         # Check if a pretrained model exists, otherwise warn
         dataset_prefix = "mnist" if args.mnist else "cifar10"
         expected_ae_path = os.path.join('results', f"{dataset_prefix}_autoencoder_best.pth")
         expected_cont_path = os.path.join('results', f"{dataset_prefix}_contrastive_best.pth")
         expected_guided_path = os.path.join('results', f"{dataset_prefix}_guided_best.pth")
         if not (os.path.exists(expected_ae_path) or os.path.exists(expected_cont_path) or os.path.exists(expected_guided_path)):
              print("Warning: --train-classifier selected without a primary training mode (--self-supervised, --contrastive, --classification-guided) in the same run, and no suitable pretrained encoder checkpoint (_autoencoder_best.pth, _contrastive_best.pth, or _guided_best.pth) found in ./results/. Classifier will train on randomly initialized encoder features.")
         else:
              print("Info: --train-classifier selected. Will attempt to load best pretrained encoder from ./results/ based on dataset name.")


    return args

def main():
    args = get_args()
    set_seed(args.seed)
    print("Args:", args)

    encoder = None 
    classifier = None 

    dataset_name = "MNIST" if args.mnist else "CIFAR10" 

    # --- Self-Supervised Autoencoder Training ---
    if args.self_supervised:
        print("\n=== STARTING SELF-SUPERVISED AUTOENCODER TRAINING ===")
        setup = setup_datasets(args, model_type='autoencoder')
        trainer = AutoencoderTrainer(args, setup)
        autoencoder_model = trainer.train() 
        encoder = autoencoder_model.encoder 
        print("=== AUTOENCODER TRAINING COMPLETE ===\n")

    # --- Contrastive Self-Supervised Training ---
    if args.contrastive:
        print("\n=== STARTING CONTRASTIVE SELF-SUPERVISED TRAINING ===")
        setup = setup_datasets(args, model_type='contrastive')
        trainer = ContrastiveTrainer(args, setup)
        encoder = trainer.train() 
        print("=== CONTRASTIVE TRAINING COMPLETE ===\n")

    # --- Classification-Guided Training ---
    if args.classification_guided:
        print("\n=== STARTING CLASSIFICATION-GUIDED TRAINING ===")
        setup = setup_datasets(args, model_type='encoder')
        trainer = ClassificationGuidedTrainer(args, setup)
        encoder, classifier = trainer.train() 
        print("=== CLASSIFICATION-GUIDED TRAINING COMPLETE ===\n")

    # --- Classifier Training (on top of pretrained encoder) ---
    if args.train_classifier:
        print("\n=== STARTING CLASSIFIER TRAINING (ON PRETRAINED ENCODER) ===")
        setup = setup_datasets(args, model_type='classifier')
        trainer = ClassifierTrainer(args, setup)
        classifier = trainer.train() 
        print("=== CLASSIFIER TRAINING COMPLETE ===\n")

    print(f"\n=== ALL REQUESTED TRAINING COMPLETED FOR {dataset_name} ===")

if __name__ == '__main__':
    main()

