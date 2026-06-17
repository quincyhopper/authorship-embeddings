import torch
import torch.nn.functional as F
import lightning as L
from model import ModelWrapper
from loss import SupConLoss
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

class ContrastiveTrainer(L.LightningModule):
    def __init__(
            self, 
            model_code, 
            lr=1e-5, 
            minibatch_size=8, 
            weight_decay=0.01, 
            warmup_steps=180,
            ):
        super().__init__()

        self.model = ModelWrapper(model_code)
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.loss_func = SupConLoss(temperature=.07)
        self.save_hyperparameters()

        self.automatic_optimization = False
        
        # Lists for storing validation embeddings for use with the KNN
        self.val_embeddings = []
        self.val_labels = []

    def configure_optimizers(self):

        # Init optimiser
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # Define total steps and warmup steps
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
            )

        return [optimizer], [scheduler]
    
    def _flatten_and_chunk(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, seq_len: int):
        # Flatten batch: [B*V, D]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attn_mask.view(-1, seq_len)

        # Chunk batch into minibatches
        minibatch_input_ids = torch.split(flat_ids, self.hparams.minibatch_size)
        minibatch_attention_mask = torch.split(flat_mask, self.hparams.minibatch_size)

        return minibatch_input_ids, minibatch_attention_mask

    def _process_minibatch(self, id, mask, anchors, start, batch_size, view_size, labels):
        rep = anchors.clone()                                          # Make copy of current batch embeddings
        end = start + id.shape[0]                                      # Calculte minibatch indices
        rep[start:end] = self.model(input_ids=id, attention_mask=mask) # Replace frozen embedding with fresh embeddings for this chunk
        rep_views = rep.view(batch_size, view_size, -1)                # Reshape back to 3D tensor: [B, V, D]
        loss = self.loss_func(rep_views, labels)                       # Compute loss
        self.manual_backward(loss)                                     # Compute gradients

        return loss.detach()

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()
        sch = self.lr_schedulers()

        input_ids, attention_mask, labels = batch
        batch_size, view_size, seq_len = input_ids.shape

        minibatch_input_ids, minibatch_attention_mask = self._flatten_and_chunk(input_ids, attention_mask, seq_len)

        # 1. Compute embeddings for entire batch with no gradients
        with torch.no_grad():
            anchors = torch.vstack([self.model(input_ids=id, attention_mask=mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # 2. Accumulate gradients and loss via minibatches
        start = 0
        for j, (id, mask) in enumerate(zip(minibatch_input_ids, minibatch_attention_mask)):
            is_last_minibatch = (j == len(minibatch_input_ids) - 1)

            # Compute loss and gradients for minibatch
            # Only sync across GPUs after the last minibatch
            if not is_last_minibatch:
                with self.trainer.strategy.block_backward_sync():
                    loss = self._process_minibatch(id, mask, anchors, start, batch_size, view_size, labels)
            else:
                loss = self._process_minibatch(id, mask, anchors, start, batch_size, view_size, labels)

            # Increment start for next iteration
            start += id.shape[0]

        # Log the training loss for WandB
        self.log('train_loss', loss, prog_bar=True, sync_dist=True) # sync_dist to average loss across all GPUs

        # Update parameters
        optim.step()
        optim.zero_grad()
        sch.step()    

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        batch_size, view_size, seq_len = input_ids.shape

        minibatch_input_ids, minibatch_attention_mask = self._flatten_and_chunk(input_ids, attention_mask, seq_len)

        # Compute embeddings (using context manager to be extra safe)
        with torch.no_grad():
            anchors = torch.vstack([self.model(input_ids=id, attention_mask=mask) for id, mask in zip(minibatch_input_ids, minibatch_attention_mask)])

        # Reshape back to 3D tensor: [B, V, D]
        rep_views = anchors.view(batch_size, view_size, -1)
        val_loss = self.loss_func(rep_views, labels)

        # Store embeddings and labels for KNN
        self.val_embeddings.append(rep_views)
        self.val_labels.append(labels)

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

        return val_loss
    
    def on_validation_epoch_end(self):
        gathered_embeddings = self.all_gather(torch.cat(self.val_embeddings, dim=0)) # (n_GPU, B, V, D)
        gathered_labels = self.all_gather(torch.cat(self.val_labels, dim=0))

        embeddings = gathered_embeddings.view(-1, gathered_embeddings.shape[-2], gathered_embeddings.shape[-1]) # (1024, 16, 512)
        labels = gathered_labels.view(-1) # (1024,)

        # only master GPU does the KNN
        if self.trainer.is_global_zero:
            n_authors, n_views, d_model = embeddings.shape
            
            all_preds = []
            all_trues = []

            # 16-fold cross-validation 
            for v in range(n_views):
                support_vecs = embeddings[:, v, :] # One text per author
                queries = torch.cat([embeddings[:, :v, :], embeddings[:, v+1:, :]], dim=1) # Extract the other 15 texts per author
                query_vecs = queries.view(-1, d_model) # Flatten (15360, 512)
                query_labels = labels.repeat_interleave(n_views - 1) # (15360,
                
                # Compute cosine similarity of queries with all supports
                support_norms = F.normalize(support_vecs, p=2, dim=-1)
                query_norms = F.normalize(query_vecs, p=2, dim=-1)
                sim_matrix = torch.matmul(query_norms, support_norms.T)
                
                # Select index of closest match in the 1024 pool
                top1_indices = torch.argmax(sim_matrix, dim=-1) # (15360,)
                preds = labels[top1_indices]

                all_preds.append(preds)
                all_trues.append(query_labels)

            # Concat results from cross-validations
            y_pred = torch.cat(all_preds).detach().cpu().numpy()
            y_true = torch.cat(all_trues).detach().cpu().numpy()

            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            self.log("val_macro_precision", report["macro avg"]["precision"], rank_zero_only=True)
            self.log("val_macro_recall",    report["macro avg"]["recall"],    rank_zero_only=True)
            self.log("val_macro_f1",        report["macro avg"]["f1-score"],  rank_zero_only=True)
            self.log("val_acc",             report["accuracy"],               rank_zero_only=True)

        self.val_embeddings.clear()
        self.val_labels.clear()