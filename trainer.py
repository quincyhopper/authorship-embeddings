import torch
import lightning as L
from math import ceil
from model import ModelWrapper
from loss import SupConLoss
from transformers import get_cosine_schedule_with_warmup

class ContrastiveTrainer(L.LightningModule):
    def __init__(self, 
                 model_code, 
                 lr=1e-5, 
                 epochs=1, 
                 minibatch_size=8,
                 weight_decay=0.01):
        super().__init__()

        self.model = ModelWrapper(model_code)
        self.lr = lr
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.weight_decay = weight_decay
        self.loss_func = SupConLoss(temperature=.07)
        self.save_hyperparameters()

        self.automatic_optimization = False

    def configure_optimizers(self):

        # Init optimiser
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay
                                      )
        
        # Define total steps and warmup steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1) # 10% of training is warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
            )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()
        sch = self.lr_schedulers()

        input_ids, attention_mask, labels = batch
        batch_size, view_size, seq_len = input_ids.shape

        # Flatten batch: [B*V, D]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Chunk batch into minibatches
        minibatch_input_ids = torch.split(flat_ids, self.hparams.minibatch_size)
        minibatch_attention_mask = torch.split(flat_mask, self.hparams.minibatch_size)

        # 1. Compute embeddings for entire batch (no activations or gradients)
        with torch.no_grad():
            anchors = torch.vstack([self.model(id, mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # 2. Re-compute embeddings for minibatch, computing loss and gradients
        start = 0
        for id, mask in zip(minibatch_input_ids, minibatch_attention_mask):
            
            # Compute loss and gradients for minibatch
            loss = self._process_minibatch(id, mask, anchors, start, batch_size, view_size, labels)

            # Increment start for next iteration
            start += id.shape[0]

        # Log the training loss for WandB
        self.log('train_loss', loss, prog_bar=True, sync_dist=True) # sync_dist to average loss across all GPUs

        # Update parameters
        optim.step()
        optim.zero_grad()
        sch.step()

    def _process_minibatch(self, id, mask, anchors, start, batch_size, view_size, labels):
        rep = anchors.clone()                           # Make copy of current batch embeddings
        end = start + id.shape[0]                       # Calculte minibatch indices
        rep[start:end] = self.model(id, mask)           # Replace frozen embedding with fresh embeddings for this chunk
        rep_views = rep.view(batch_size, view_size, -1) # Reshape back to 3D tensor: [B, V, D]
        loss = self.loss_func(rep_views, labels)        # Compute loss
        self.manual_backward(loss)                      # Compute gradients

        return loss.detach()

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, labels = batch
        batch_size, view_size, seq_len = input_ids.shape

        # Flatten batch: [B*V, D]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Chunk batch into minibatches
        minibatch_input_ids = torch.split(flat_ids, self.hparams.minibatch_size)
        minibatch_attention_mask = torch.split(flat_mask, self.hparams.minibatch_size)

        # Compute embeddings
        anchors = torch.vstack([self.model(id, mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # Reshape back to 3D tensor: [B, V, D]
        rep_views = anchors.view(batch_size, view_size, -1)

        # Calculate loss 
        val_loss = self.loss_func(rep_views, labels)

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

        return val_loss
