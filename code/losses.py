import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.
    (Based on SimCLR paper: https://arxiv.org/abs/2002.05709)
    """
    def __init__(self, temperature=0.1, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # Use sum reduction

    def forward(self, z_i, z_j):
        """
        Calculates NT-Xent loss.
        Args:
            z_i: Tensor of shape [batch_size, feature_dim] for view 1.
            z_j: Tensor of shape [batch_size, feature_dim] for view 2.
        Returns:
            Loss tensor.
        """
        batch_size = z_i.shape[0]
        feature_dim = z_i.shape[1]

        # Normalize features
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate features: [2*batch_size, feature_dim]
        representations = torch.cat([z_i, z_j], dim=0)

        # Calculate similarity matrix: [2*batch_size, 2*batch_size]
        # Cosine similarity: dot product of normalized vectors
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # Create positive mask: identifies pairs (i, i+N) and (i+N, i)
        # Exclude self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1) # Shape: [2N, 2N-1]

        # Get positive samples (the other view of the same image)
        # Positive pair for z_i is z_j, positive pair for z_j is z_i
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size - 1), # Similarities (z_i, z_j)
            torch.diag(similarity_matrix, -batch_size + 1) # Similarities (z_j, z_i)
        ], dim=0) # Shape: [2N]

        # Scale similarities by temperature
        logits = similarity_matrix / self.temperature
        positives = positives / self.temperature

        # Create labels for cross-entropy loss
        # For each row in logits (similarity of one view with all others),
        # the positive key is at a specific index.
        # For the first N rows (z_i), the positive key (z_j) is at index N-1 + k (relative to the non-diagonal view)
        # For the next N rows (z_j), the positive key (z_i) is at index k
        # Since we removed the diagonal, the indices shift.
        # The positive sample for row k (original z_i[k]) is now at index k + N - 1 in the flattened view.
        # The positive sample for row k+N (original z_j[k]) is now at index k in the flattened view.
        labels = torch.arange(batch_size, device=self.device, dtype=torch.long)
        labels = torch.cat([labels + batch_size - 1, labels], dim=0)


        # Calculate cross-entropy loss
        # Logits shape: [2N, 2N-1], Labels shape: [2N]
        # The labels indicate the column index of the positive sample for each row.
        loss = self.criterion(logits, labels)

        # Normalize loss by the number of samples (2 * batch_size)
        loss = loss / (2 * batch_size)
        return loss 