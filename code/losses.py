import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) from SimCLR
    
    This can be used as a drop-in criterion like nn.L1Loss or nn.CrossEntropyLoss
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1: First batch of projections [batch_size, projection_dim]
            z2: Second batch of projections [batch_size, projection_dim]
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate embeddings from both augmentations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                              representations.unsqueeze(0), 
                                              dim=2)
        
        # Remove diagonal entries (self-similarity)
        sim_i_j = torch.diag(similarity_matrix, batch_size)  # Similarity between i and i+batch_size
        sim_j_i = torch.diag(similarity_matrix, -batch_size)  # Similarity between i+batch_size and i
        
        # Positives are the similarities between augmented pairs
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Create mask to identify the negative samples for each anchor
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=device)
        
        # Remove self-similarity and positives from the similarity matrix
        # to isolate only the negatives
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)
        
        # Compute logits: positive / temperature
        logits = torch.cat([positives.unsqueeze(1) / self.temperature, 
                          negatives / self.temperature], dim=1)
        
        # Labels always point to the positives (index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss 