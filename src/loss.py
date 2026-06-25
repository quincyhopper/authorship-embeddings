import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: tensor of shape [B, V, D]
            labels: tensor of shape [B]

        Returns:
            A loss scalar
        """
        if len(features.shape) < 3:
            raise ValueError('Features needs to be [B, V, D]',
                             '3 dimension are required')

        device = features.device
        view_size = features.shape[1]

        # Make labels 2D tensor: [B, 1]
        labels = labels.contiguous().view(-1, 1)

        # Flatten embeddings: [B*V, D]
        batch_embs = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Normalise embeddings for cosine similarity
        batch_embs = F.normalize(batch_embs)

        # Compute similarity matrix: [B*V, B*V]
        logits = torch.matmul(batch_embs, batch_embs.T) / self.temperature

        # Change same-author logits to -INF
        logits_mask = torch.eye(logits.shape[0], device=device).bool() # [B*V, B*V] identity matrix
        logits = logits.masked_fill(logits_mask, -float('inf'))        # Replace diagonal logits with -INF

        # Compute log probabilities 
        log_prob = F.log_softmax(logits, dim=1)
        
        # Make author-level mask (1 if authors are same)
        author_mask = torch.eq(labels, labels.T).float()       # [B, B]
        author_mask = author_mask.repeat(view_size, view_size) # [B*V, B*V]

        # Make diagonal zero so that same-texts do not count as a positive case
        author_mask = author_mask.masked_fill(logits_mask, 0)
        
        # Keep log probs where authors are the same (except diagonal) and set all others to zero
        # Using torch.where to avoid -inf * 0 errors
        positive_log_probs = torch.where(author_mask == 1, log_prob, torch.zeros_like(log_prob))

        mask_pos_pairs = author_mask.sum(1).clamp(min=1e-6) # Number of positive cases per text (should equal view_size-1)
        mean_log_probs_pos = positive_log_probs.sum(1) / mask_pos_pairs

        # Compute loss
        loss = - mean_log_probs_pos.mean()

        return loss