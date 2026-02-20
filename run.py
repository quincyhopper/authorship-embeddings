import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer
from trainer import ContrastiveTrainer
from data_builder import AuthorshipDataModule
torch.set_float32_matmul_precision('medium')

MODEL_CODE = 'roberta-large'
DATA_PATH = 'data/blogtext_16.csv'
MAX_EPOCHS = 20
GLOBAL_BATCH_SIZE = 1024
VIEW_SIZE = 16
MAX_SEQ_LEN = 512
MINIBATCH_SIZE = 32

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

    # Read data
    df = pd.read_csv(DATA_PATH)

    # Init tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE)
    
    # Init data loaders
    data_module = AuthorshipDataModule(df, 
                                       tokenizer=tokenizer, 
                                       batch_size=GLOBAL_BATCH_SIZE, 
                                       view_size=VIEW_SIZE,
                                       max_seq_len=MAX_SEQ_LEN)

    # Init Lightning Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        strategy='auto',
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