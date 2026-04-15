import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from trainer import ContrastiveTrainer
from data_builder import AuthorshipDataModule

MODEL_CODE = 'roberta-large'
MAX_STEPS = 3000
GLOBAL_BATCH_SIZE = 1024
VIEW_SIZE = 16
MAX_SEQ_LEN = 512
MINIBATCH_SIZE = 48
LR = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 180
NUM_WORKERS = 8

TRAIN_PATH = [
    'data/reddit_train.parquet',
    'data/twitter_train.parquet',
    'data/gutenberg_train.parquet',
    'data/blogtext_train.parquet'
]
VAL_PATH = 'data/reddit_val.parquet'

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    # Init model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="star-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        every_n_train_steps=250
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
    data_module = AuthorshipDataModule(train_path=TRAIN_PATH,
                                       val_path=VAL_PATH, 
                                       batch_size=GLOBAL_BATCH_SIZE, 
                                       view_size=VIEW_SIZE,
                                       max_seq_len=MAX_SEQ_LEN,
                                       num_workers=NUM_WORKERS
                                       )

    # Init Lightning Trainer
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=4,
        strategy='ddp_find_unused_parameters_true',
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1 # Perform validation once per epoch
    )

    # Init trainer
    model = ContrastiveTrainer(MODEL_CODE, 
                               lr=LR, 
                               minibatch_size=MINIBATCH_SIZE,
                               weight_decay=WEIGHT_DECAY,
                               warmup_steps=WARMUP_STEPS
                               )

    # 3. Train
    trainer.fit(model, data_module)