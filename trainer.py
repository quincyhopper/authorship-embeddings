import torch
import lightning as L
from math import ceil
from model import ModelWrapper
from loss import SupConLoss

class ContrastiveTrainer(L.LightningModule):
    def __init__(self, model_code, lr=1e-5, epochs=1, minibatch_size=8):
        super().__init__()

        self.save_hyperparameters()
        self.model = ModelWrapper(model_code)
        self.loss_func = SupConLoss()

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs, eta_min=1e-6
            )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        optim = self.optimizers()

        input_ids, attention_mask, labels = batch
        batch_size, view_size, seq_len = input_ids.shape

        # Flatten batch: [B*V, D]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Chunk batch into minibatches
        n = int(ceil((batch_size * view_size) / self.hparams.minibatch_size)) # Calculate number of minibatches
        minibatch_input_ids = torch.chunk(flat_ids, chunks=n)
        minibatch_attention_mask = torch.chunk(flat_mask, chunks=n)

        # 1. Compute embeddings for entire batch (no activations or gradients)
        with torch.no_grad():
            anchors = torch.vstack([self.model(id, mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # 2. Re-compute embeddings for minibatch, computing loss and gradients
        for j, (id, mask) in enumerate(zip(minibatch_input_ids, minibatch_attention_mask)):
            
            # Make a copy of the current batch embeddings
            rep = anchors.clone()

            # Determine indices of current batch
            start = j * self.hparams.minibatch_size
            end = (j+1) * self.hparams.minibatch_size

            # Replace frozen embedding with fresh embeddings for this chunk
            rep[start:end] = self.model(id, mask)

            # Reshape back to 3D tensor: [B, V, D]
            rep_views = rep.view(batch_size, view_size, -1)
            
            loss = self.loss_func(rep_views, labels)
            self.manual_backward(loss)

        with torch.no_grad():
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        optim.step()
        optim.zero_grad()

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, labels = batch
        batch_size, view_size, seq_len = input_ids.shape

        # Flatten batch: [B*V, D]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Chunk batch into minibatches
        n = int(ceil((batch_size * view_size) / self.hparams.minibatch_size)) # Calculate number of minibatches
        minibatch_input_ids = torch.chunk(flat_ids, chunks=n)
        minibatch_attention_mask = torch.chunk(flat_mask, chunks=n)

        # Compute embeddings
        anchors = torch.vstack([self.model(id, mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # Reshape back to 3D tensor: [B, V, D]
        rep_views = anchors.view(batch_size, view_size, -1)

        # Calculate loss 
        val_loss = self.loss_func(rep_views, labels)

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

        return val_loss
