import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split, Dataset
from models import MNISTEncoder, CIFAR10Encoder, MNISTAutoencoder, CIFAR10Autoencoder

def plot_tsne(model, dataloader, device, dataset_name=None):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to over over data for which you wish to compute projections
    device - cuda or cpu (as a string)
    dataset_name - Optional name of dataset for file naming
    '''
    model.eval()
    
    images_list = []
    labels_list = []
    latent_list = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            #approximate the latent space from data
            latent_vector = model(images)
            
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)
    
    # Determine dataset name for file paths
    ds_name = dataset_name if dataset_name else "unknown"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
    plt.colorbar(scatter)
    plt.title(f't-SNE of Latent Space ({ds_name})')
    plt.savefig(f'results/{ds_name}_latent_tsne.png')
    plt.close()
    
    #plot image domain tsne
    tsne_image = TSNE(n_components=2, random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)  
    plt.colorbar(scatter)
    plt.title(f't-SNE of Image Space ({ds_name})')
    plt.savefig(f'results/{ds_name}_image_tsne.png')
    plt.close()

# --- Contrastive Augmentations ---
class ContrastiveTransform:
    """Applies two separate augmentation pipelines to the same image."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]

def get_contrastive_augmentations(img_size, is_mnist=False):
    """Returns SimCLR-style augmentations for contrastive learning."""
    # Color jitter parameters differ slightly for MNIST (grayscale) vs CIFAR
    if is_mnist:
        # MNIST is grayscale, less color jitter needed. Focus on geometric.
        # SimCLR used ColorJitter even on ImageNet, but strength might vary.
        # Let's use mild jitter and stronger geometric transforms.
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4) # No saturation/hue for grayscale
        gaussian_blur = transforms.GaussianBlur(kernel_size=int(0.1 * img_size) | 1, sigma=(0.1, 2.0))

    else: # CIFAR10
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        gaussian_blur = transforms.GaussianBlur(kernel_size=int(0.1 * img_size) | 1, sigma=(0.1, 2.0))


    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)), # Stronger crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2 if not is_mnist else 0.0), # Don't grayscale MNIST
        transforms.RandomApply([gaussian_blur], p=0.5 if not is_mnist else 0.1), # Less blur for MNIST?
        transforms.ToTensor(),
        # Normalization applied later in setup_datasets
    ])
    return transform

# --- Dataset Wrapper for Contrastive Learning ---
class ContrastiveDatasetWrapper(Dataset):
    """Wraps a dataset to apply contrastive transforms and return (view1, view2, target)."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = ContrastiveTransform(transform) # Applies transform twice

    def __getitem__(self, index):
        img, target = self.dataset[index]
        view1, view2 = self.transform(img) # Get two augmented views
        return view1, view2, target

    def __len__(self):
        return len(self.dataset)


# --- Existing plot functions (keep as is) ---
def plot_reconstructions(originals, reconstructions, epoch, result_dir, dataset_name):
    # ... (keep existing implementation)
    num_images = originals.shape[0]
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(originals[i].permute(1, 2, 0).squeeze(), cmap='gray' if originals.shape[1]==1 else None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == num_images // 2: ax.set_title("Original Images")

        # Reconstructed images
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructions[i].permute(1, 2, 0).squeeze(), cmap='gray' if reconstructions.shape[1]==1 else None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == num_images // 2: ax.set_title("Reconstructed Images")

    plt.suptitle(f'Epoch {epoch} Reconstructions')
    save_path = os.path.join(result_dir, f"{dataset_name.lower()}_reconstructions_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()


def plot_tsne(features, labels, epoch, result_dir, dataset_name, trainer_type):
    # ... (keep existing implementation)
    print("Generating t-SNE plot...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title(f't-SNE of Latent Space - Epoch {epoch} ({trainer_type})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(max(labels)+1))
    save_path = os.path.join(result_dir, f"{dataset_name.lower()}_{trainer_type.lower()}_tsne_epoch_{epoch}.png")
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")
    plt.close()


# --- Refactored setup_datasets ---
def setup_datasets(args, model_type='encoder'):
    """
    Sets up datasets, dataloaders, and model based on args and model_type.
    model_type options: 'autoencoder', 'encoder', 'classifier', 'contrastive'
    """
    if args.mnist:
        dataset_config = {
            'name': 'MNIST',
            'dataset_class': MNIST,
            'model_class': MNISTAutoencoder if model_type == 'autoencoder' else MNISTEncoder,
            'mean': (0.1307,),
            'std': (0.3081,),
            'img_size': 28,
            'num_classes': 10,
            'is_mnist': True
        }
    else:
        dataset_config = {
            'name': 'CIFAR10',
            'dataset_class': CIFAR10,
            'model_class': CIFAR10Autoencoder if model_type == 'autoencoder' else CIFAR10Encoder,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.247, 0.243, 0.261),
            'img_size': 32,
            'num_classes': 10,
            'is_mnist': False
        }

    img_size = dataset_config['img_size']
    is_mnist = dataset_config['is_mnist']
    normalize = transforms.Normalize(mean=dataset_config['mean'], std=dataset_config['std'])

    # Base transform for validation/test (and non-contrastive training)
    base_transform_list = [transforms.ToTensor(), normalize]
    if not is_mnist: # Resize CIFAR for consistency if needed, MNIST is already 28x28
         if img_size != 32: # Example if you wanted different size encoders
              base_transform_list.insert(0, transforms.Resize(img_size))
    base_transform = transforms.Compose(base_transform_list)


    # Training transforms depend on the model type
    if model_type == 'contrastive':
        # Use SimCLR-style augmentations, normalization is applied *after* other transforms
        contrastive_augs = get_contrastive_augmentations(img_size, is_mnist)
        # Apply normalization within the contrastive wrapper or after getting views?
        # Let's apply it to each view after the base contrastive augs.
        train_transform = transforms.Compose([
             contrastive_augs, # This generates the list of views
             transforms.Lambda(lambda views: [normalize(view) for view in views]) # Normalize each view
        ])
        print("[Dataset] Using Contrastive Augmentations for Training.")
    else:
        # Standard augmentations for autoencoder/classifier/guided
        augmentations = []
        if not is_mnist: # Add standard CIFAR augmentations
            augmentations.extend([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                # Add more mild augmentations if desired
            ])
        # Combine augmentations with ToTensor and Normalize
        train_transform = transforms.Compose(
            augmentations + [transforms.ToTensor(), normalize]
        )
        print("[Dataset] Using Standard Augmentations for Training.")


    # Create datasets
    full_train_dataset = dataset_config['dataset_class'](
        root=args.data_path, train=True, download=True,
        # Apply transform directly unless contrastive
        transform=None if model_type == 'contrastive' else train_transform
    )

    # Apply contrastive wrapper if needed (needs untransformed dataset)
    if model_type == 'contrastive':
         # The ContrastiveDatasetWrapper needs the *base* augmentations,
         # normalization will be applied by the DataLoader transform.
         base_contrastive_augs = get_contrastive_augmentations(img_size, is_mnist)
         contrastive_wrapper_transform = transforms.Compose([
              base_contrastive_augs, # Generates list of PIL images
              transforms.Lambda(lambda views: [normalize(transforms.ToTensor()(view)) for view in views]) # ToTensor+Normalize each view
         ])
         # Re-create dataset with only download=True, no transform yet
         base_train_dataset = dataset_config['dataset_class'](
              root=args.data_path, train=True, download=True, transform=None
         )
         full_train_dataset = ContrastiveDatasetWrapper(base_train_dataset, contrastive_wrapper_transform)


    # Split train/validation
    val_size = 5000
    train_size = len(full_train_dataset) - val_size
    # Ensure val_size is not larger than dataset
    if val_size >= len(full_train_dataset):
         print(f"Warning: Validation size ({val_size}) >= dataset size ({len(full_train_dataset)}). Adjusting validation size.")
         val_size = int(len(full_train_dataset) * 0.1) # Use 10% for validation
         train_size = len(full_train_dataset) - val_size

    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Create test dataset
    test_dataset = dataset_config['dataset_class'](
        root=args.data_path, train=False, download=True, transform=base_transform
    )

    # Create dataloaders
    # Note: Contrastive wrapper handles transform, so DataLoader doesn't need it again
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Validation loader needs careful handling for contrastive vs others
    if model_type == 'contrastive':
         # Validation for contrastive: need two views for loss, one view + target for probe
         # Option 1: Use the same contrastive wrapper for val loader
         # Option 2: Use base_transform for probe eval, need separate logic for contrastive loss eval
         # Let's use Option 1 for simplicity, ContrastiveTrainer._validate_epoch handles it
         val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
         # For non-contrastive, val_subset already has train_transform applied by split.
         # This is usually NOT desired. Validation should use base_transform.
         # Re-create val_subset with base_transform.
         val_indices = val_subset.indices
         clean_val_dataset = dataset_config['dataset_class'](
              root=args.data_path, train=True, download=True, transform=base_transform # Use base transform!
         )
         val_subset_clean = torch.utils.data.Subset(clean_val_dataset, val_indices)
         val_loader = DataLoader(val_subset_clean, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model instance
    # For contrastive, model_class is Encoder. For others, it's set based on type.
    model = dataset_config['model_class'](args.latent_dim).to(args.device)

    # Construct the return dictionary
    setup = {
        'dataset_name': dataset_config['name'],
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'is_mnist': dataset_config['is_mnist'],
        'img_size': dataset_config['img_size'],
        'num_classes': dataset_config['num_classes'],
        'model': model # This is the primary model (Autoencoder or Encoder)
    }

    # Add specific references if needed by trainers (though they often derive from self.model)
    if model_type == 'autoencoder':
        setup['encoder'] = model.encoder
        setup['decoder'] = model.decoder
    elif model_type in ['encoder', 'classifier', 'guided', 'contrastive']:
        setup['encoder'] = model # Encoder is the main model

    return setup

def get_result_dir(args, training_type):
    """
    Create and return an appropriate result directory based on 
    training type and dataset.
    
    Args:
        args: Command line arguments
        training_type: 'self_supervised' or 'classification_guided'
        
    Returns:
        str: Path to the result directory
    """
    dataset_name = "mnist" if args.mnist else "cifar10"
    result_dir = os.path.join('results', training_type, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir