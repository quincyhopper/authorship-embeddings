import torch
from data_builder import build_supervised_dataset
from transformers import AutoTokenizer
from model import ModelWrapper
from trainer import ContrastiveTrainer
import pandas as pd
from tqdm import tqdm

MODEL_CODE = 'prajjwal1/bert-tiny'
DATA_PATH = 'src/data/blogtext.csv'
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_BATCH_SIZE = 64
VIEW_SIZE = 3
MAX_LEN = 512
MINIBATCH_SIZE = 8
DOCS_TO_SAMPLE = 512

tokeniser = AutoTokenizer.from_pretrained(MODEL_CODE)
model = ModelWrapper(MODEL_CODE).to(DEVICE)

train_loader = build_supervised_dataset(
    pd.read_csv(DATA_PATH),
    tokeniser=tokeniser,
    batch_size=GLOBAL_BATCH_SIZE,
    view_size=VIEW_SIZE,
    max_seq_len=MAX_LEN,
    max_docs=DOCS_TO_SAMPLE
)

train_model = ContrastiveTrainer(model,
                                 device=DEVICE,
                                 learning_rate=1e-4,
                                 weight_decay=None, 
                                 epochs=EPOCHS, 
                                 minibatch_size=MINIBATCH_SIZE
                                 )

def train(trainer, train_loader, epochs):
    trainer.model.train()

    for epoch in range(epochs):

        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):

            loss = trainer.train(batch)
            epoch_loss += loss

        print(f"Epoch [{epoch+1}/{epochs}] | Train loss {epoch_loss}")


if __name__ == "__main__":
    train(train_model, train_loader, EPOCHS)