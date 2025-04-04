import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

# need to fix the paths
MNIST_CHECKPOINT_PATH = "./mnist_autoencoder.pth"
CIFAR10_CHECKPOINT_PATH = "./cifar10_autoencoder.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from models import MNISTAutoencoder, CIFAR10Autoencoder

# load autoencoder
def load_autoencoder(model_class, checkpoint_path, device):
    model = model_class().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:

        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    return model

# get random samples
def get_random_samples(dataset, num_samples = 5):
    indices = random.sample(range(len(dataset)), num_samples)
    imgs, labels = [], []

    for i in indices:
        img, label = dataset[i]
        imgs.append(img)
        labels.append(label)

    return torch.stack(imgs), torch.tensor(labels)

# plot original and reconstructed images
def plot_original_vs_recon(
    original_batch: torch.Tensor,
    recon_batch: torch.Tensor,
    labels: list,
    indices: list,
    title_prefix: str,
    is_gray: bool = False
):
    n = original_batch.size(0)
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    fig.suptitle(f"{title_prefix} Autoencoder Reconstructions", fontsize=14)

    for i in range(n):
        original_img = original_batch[i]
        recon_img = recon_batch[i]

        if is_gray:
            original_img = original_img.squeeze(0)
            recon_img = recon_img.squeeze(0)
            cmap = 'gray'
        else:
            original_img = original_img.permute(1, 2, 0)
            recon_img = recon_img.permute(1, 2, 0)
            cmap = None

        # plot original and reconstructed images
        axes[0, i].imshow(original_img, cmap=cmap)
        axes[0, i].set_title(f"Original\nLabel: {labels[indices[i]]}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon_img, cmap=cmap)
        axes[1, i].set_title(f"Reconstructed\nLabel: {labels[indices[i]]}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{title_prefix}_reconstructions.png", dpi=300)
    plt.close()


def main():
    # set random seeds for reproducibility
    SEED = 22071999

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE == 'cuda':
        torch.manual_seed_all(SEED)

    # CIAFR10 - propably need to fix the data path
    cifar_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    cifar_model = load_autoencoder(CIFAR10Autoencoder, CIFAR10_CHECKPOINT_PATH, DEVICE)

    # get random samples
    cifar_random_imgs, cifar_imgs, cifar_labels = get_random_samples(cifar_data)
    cifar_imgs = cifar_imgs.to(DEVICE)

    with torch.no_grad():
        cifar_recon = cifar_model(cifar_imgs)

    plot_original_vs_recon(
        cifar_imgs,
        cifar_recon,
        cifar_labels,
        cifar_random_imgs,
        "CIFAR10",
        is_gray=False
    )

    # MNIST - propably need to fix the data path
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_model = load_autoencoder(MNISTAutoencoder, MNIST_CHECKPOINT_PATH, DEVICE)

    # get random samples
    mnist_random_imgs, mnist_imgs, mnist_labels = get_random_samples(mnist_data)
    mnist_imgs = mnist_imgs.to(DEVICE)

    with torch.no_grad():
        mnist_recon = mnist_model(mnist_imgs)

    plot_original_vs_recon(
        mnist_imgs,
        mnist_recon,
        mnist_labels,
        mnist_random_imgs,
        "MNIST",
        is_gray=True
    )

    # save pair of orig and recon images for interpolation
    image_index_in_batch = 0

    orig_tensor = mnist_imgs[image_index_in_batch].cpu() 
    recon_tensor = mnist_recon[image_index_in_batch].cpu()

    orig_tensor_clamped = torch.clamp(orig_tensor, 0, 1)
    recon_tensor_clamped = torch.clamp(recon_tensor, 0, 1)


    orig_np = orig_tensor_clamped.squeeze(0).numpy()  
    recon_np = recon_tensor_clamped.squeeze(0).numpy()

    orig_8u = (orig_np * 255).astype(np.uint8)
    recon_8u = (recon_np * 255).astype(np.uint8)

    orig_pil = Image.fromarray(orig_8u, mode='L')
    recon_pil = Image.fromarray(recon_8u, mode='L')

 
    original_dataset_index = mnist_random_imgs[image_index_in_batch]
    label = mnist_labels[image_index_in_batch].item() 

    save_dir = "imagesForLinearInterpolation"
    os.makedirs(save_dir, exist_ok=True)

    orig_filename = f"mnist_orig_idx{original_dataset_index}_label{label}.png"
    recon_filename = f"mnist_recon_idx{original_dataset_index}_label{label}.png"
    orig_path = os.path.join(save_dir, orig_filename)
    recon_path = os.path.join(save_dir, recon_filename)

    orig_pil.save(orig_path)
    recon_pil.save(recon_path)

if __name__ == "__main__":
    main()
    
    