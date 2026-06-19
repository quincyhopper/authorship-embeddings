import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from trainer import ContrastiveTrainer
from data_builder import AuthorshipDataModule

MODEL_CODE = 'roberta-large'
MAX_STEPS = 450
NUM_DEVICES = 4
PER_GPU_BATCH_SIZE = 1024
GLOBAL_BATCH_SIZE = NUM_DEVICES * PER_GPU_BATCH_SIZE # 4096
VIEW_SIZE = 16
MAX_SEQ_LEN = 512
MINIBATCH_SIZE = 48
LR = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 27
NUM_WORKERS = 8

TRAIN_PATH = ['data/train.parquet']
VAL_PATH = ['data/val.parquet']

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    RUN_NAME = 'sanity_check'

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename=RUN_NAME,
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
    )

    # Init loggers
    wandb_logger = WandbLogger(project="authorship-embeddings", name=RUN_NAME)
    csv_logger = CSVLogger(save_dir='log/', name=RUN_NAME)
    
    # Init data loaders
    data_module = AuthorshipDataModule(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH, 
        batch_size=GLOBAL_BATCH_SIZE, 
        view_size=VIEW_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_workers=NUM_WORKERS,
        content_masking=False,
        masking_threshold=None,
    )

    # Init Lightning Trainer
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=NUM_DEVICES,
        strategy='ddp_find_unused_parameters_true',
        precision="16-mixed",
        callbacks=[early_stopping_callback, checkpoint_callback],
        enable_checkpointing=True,
        logger=[wandb_logger, csv_logger],
        log_every_n_steps=1,
        check_val_every_n_epoch=1, # Perform validation once per epoch
        use_distributed_sampler=False,
    )

    # Init trainer
    model = ContrastiveTrainer(
        MODEL_CODE, 
        lr=LR, 
        minibatch_size=MINIBATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
    )

    # 3. Train
    trainer.fit(model, data_module)