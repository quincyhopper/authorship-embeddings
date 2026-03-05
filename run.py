import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from trainer import ContrastiveTrainer
from data_builder import AuthorshipDataModule
torch.set_float32_matmul_precision('medium')

MODEL_CODE = 'roberta-large'
MAX_EPOCHS = 1
GLOBAL_BATCH_SIZE = 256
VIEW_SIZE = 16
MAX_SEQ_LEN = 512
MINIBATCH_SIZE = 48

if __name__ == "__main__":

    # Init model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="star-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Init early stoppping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    # Init logger
    wandb_logger = WandbLogger(
        project="authorship-embeddings"
    )
    
    # Init data loaders
    data_module = AuthorshipDataModule(train_path='data/train_chunks.parquet',
                                       val_path='data/val_chunks.parquet', 
                                       batch_size=GLOBAL_BATCH_SIZE, 
                                       view_size=VIEW_SIZE,
                                       max_seq_len=MAX_SEQ_LEN,
                                       num_workers=1
                                       )

    # Init Lightning Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=4,
        strategy='ddp_find_unused_parameters_true',
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=1.0
    )

    # Init trainer
    model = ContrastiveTrainer(MODEL_CODE, 
                               lr=1e-5, 
                               epochs=MAX_EPOCHS, 
                               minibatch_size=MINIBATCH_SIZE,
                               weight_decay=0.01,
                               )

    # 3. Train
    trainer.fit(model, data_module)