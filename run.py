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
GLOBAL_BATCH_SIZE = 256
VIEW_SIZE = 8
MAX_SEQ_LEN = 512
MINIBATCH_SIZE = 48

if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    # Init Data and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE)
    model = ContrastiveTrainer(MODEL_CODE, lr=1e-5, epochs=MAX_EPOCHS, minibatch_size=MINIBATCH_SIZE)

    data_module = AuthorshipDataModule(df, tokenizer=tokenizer, 
                                       batch_size=GLOBAL_BATCH_SIZE, 
                                       view_size=VIEW_SIZE,
                                       max_seq_len=MAX_SEQ_LEN)

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

    # Define logger
    wandb_logger = WandbLogger(
        project="authorship-embeddings"
    )

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

    # 3. Train
    trainer.fit(model, data_module)