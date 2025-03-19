import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

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