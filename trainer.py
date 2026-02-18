import torch.nn as nn
import torch
import random
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from loss import SupConLoss
from math import ceil

class ContrastiveTrainer(nn.Module):
    def __init__(self, model, device, learning_rate, weight_decay, epochs, minibatch_size):
        super().__init__()

        self.device = device
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_training_steps = epochs
        self.minibatch_size = minibatch_size

        self.optim = AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_func = SupConLoss()
        self.scaler = GradScaler()

    def train(self, batch):

        input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
        batch_size, view_size, seq_len = input_ids.shape

        self.optim.zero_grad()

        # Flatten batch: [B*V, D]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Calculate number of minibatches
        if (batch_size * view_size) % self.minibatch_size != 0:
            raise ValueError(f"Global batch size * view size must be divisible by minibatch size. {batch_size*view_size} is not divisible by {self.minibatch_size}.")
        n = int(ceil(batch_size * view_size / self.minibatch_size))

        # Chunk batch into minibatches
        minibatch_input_ids = torch.chunk(flat_ids, chunks=n)
        minibatch_attention_mask = torch.chunk(flat_mask, chunks=n)

        # 1. Compute embeddings for entire batch (no activations or gradients)
        with torch.no_grad():
            with autocast(device_type='cuda'):
                anchors = torch.vstack([self.model(id, mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # 2. Re-compute embeddings for minibatch, computing loss and gradients
        for j, (id, mask) in enumerate(zip(minibatch_input_ids, minibatch_attention_mask)):
            with autocast(device_type='cuda'):
            
                # Make a copy of the current batch embeddings
                rep = anchors.clone()

                # Determine indices of current batch
                start = j * self.minibatch_size
                end = (j+1) * self.minibatch_size

                # Replace frozen embedding with fresh embeddings for this chunk
                rep[start:end] = self.model(id, mask)

                # Reshape back to 3D tensor: [B, V, D]
                rep_views = rep.view(batch_size, view_size, -1)

                # Calculate loss using ALL documents in global batch
                loss = self.loss_func(rep_views, labels)

            # Accumulate gradients on fresh chunk
            self.scaler.scale(loss).backward()

        self.scaler.step(self.optim)
        self.scaler.update()

        return loss.item()
    
    def eval(self, batch):

        # Don't track gradients
        self.model.eval()
        with torch.no_grad():
            input_ids, attention_mask, labels = batch
            batch_size, view_size, seq_len = input_ids.shape

            # Flatten batch: [B*V, D]
            flat_ids = input_ids.view(-1, seq_len)
            flat_mask = attention_mask.view(-1, seq_len)

            # Calculate number of minibatches
            n = int(ceil(batch_size * view_size / self.minibatch_size))

            # Chunk batch into minibatches
            minibatch_input_ids = torch.chunk(flat_ids, chunks=n)
            minibatch_attention_mask = torch.chunk(flat_mask, chunks=n)

            # Generate embeddings for entire batch: [B*V, D]
            all_embeddings = torch.vstack([self.model(id, mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

            # Reshape back to 3D tensor: [B, V, D]
            rep_views = all_embeddings.reshape(batch_size, view_size, -1)

            # Compute SupCon loss
            loss = self.loss_func(rep_views, labels)

            # --- Calculate accuracy ---
            # randomly sample 2 views (two texts from each author)
            # NOTE: probably replace this or remove it entirely
            samples = random.sample(range(rep_views.shape[1]), k=2)
            anchors = rep_views[:, samples[0], :]  # [B, D]
            replicas = rep_views[:, samples[1], :] # [B, D]

            # Compute similarity between all anchors and replicas: [B, B]
            logits = torch.matmul(anchors, replicas.T)

            # Positive pairs are the diagonal
            target = torch.arange(batch_size)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == target).float().mean()

        # Switch back to train mode
        self.model.train()

        return loss.item(), accuracy.item()