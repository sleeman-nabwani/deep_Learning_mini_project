import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# need to fix path
MNIST_CHECKPOINT_PATH = "checkpoints/mnist_autoencoder.pth"
img1_path = "imagesForLinearInterpolation/mnist_orig_idx0_label0.png"
img2_path = "imagesForLinearInterpolation/mnist_recon_idx0_label0.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

interpolation_steps = 10

from models import MNISTAutoencoder

def load_autoencoder(model_class, checkpoint_path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def main():
    # set random seeds for reproducibility
    SEED = 22071999

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    # load autoencoder
    autoencoder = load_autoencoder(MNISTAutoencoder, MNIST_CHECKPOINT_PATH, DEVICE)

    # load images
    transform = transforms.ToTensor()    

    img1_converted = transform(Image.open(img1_path)).convert("L")
    img2_converted = transform(Image.open(img2_path)).convert("L")

    x1 = transform(img1_converted).unsqueeze(0).to(DEVICE)
    x2 = transform(img2_converted).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z1 = autoencoder.encode(x1)
        z2 = autoencoder.encode(x2)

    # interpolation
    alphas = torch.linspace(0, 1, interpolation_steps)
    interpolated_outputs = []

    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            x_recon = autoencoder.decode(z)
            interpolated_outputs.append(x_recon.squeeze(0))
    
    # save interpolated images
    fig, axes = plt.subplots(1, interpolation_steps, figsize=(2 * interpolation_steps, 2))
    fig.suptitle("MNIST Linear Interpolation (Single Pair)", fontsize=14)


    for i, recon_img in enumerate(interpolated_outputs):
        recon_img_np = recon_img.cpu().squeeze(0).numpy()  # shape: [H,W]
        axes[i].imshow(recon_img_np, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"α={alphas[i]:.2f}")

    plt.tight_layout()
    os.makedirs("imagesForLinearInterpolation", exist_ok=True)
    save_file = os.path.join("imagesForLinearInterpolation", "mnist_linear_interp_single_pair.png")
    plt.savefig(save_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
