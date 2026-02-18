import torch
import pandas as pd
from tqdm import tqdm 
from data_builder import build_supervised_dataset, filter_authors
from transformers import AutoTokenizer
from model import ModelWrapper
from trainer import ContrastiveTrainer
from sklearn.model_selection import train_test_split

MODEL_CODE = 'prajjwal1/bert-tiny'
DATA_PATH = 'data/blogtext.csv'
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_BATCH_SIZE = 256
VIEW_SIZE = 3
MAX_LEN = 32
MINIBATCH_SIZE = 8

def train(trainer, train_loader, epochs):
    trainer.model.train()

    for epoch in range(epochs):

        epoch_loss = 0.0

        for batch in tqdm(train_loader):

            loss = trainer.train(batch)
            epoch_loss += loss

        print(f"Epoch [{epoch+1}/{epochs}] | Train loss {epoch_loss/len(train_loader)}")

if __name__ == "__main__":

    # Initialise tokeniser and model
    tokeniser = AutoTokenizer.from_pretrained(MODEL_CODE)
    model = ModelWrapper(MODEL_CODE).to(DEVICE)

    # Prepare data and loader
    print(f"Initialising loader")
    df = pd.read_csv(DATA_PATH)        # 681,284 rows
    df = filter_authors(df, VIEW_SIZE) # Remove authors that don't have enough texts

    train_df, val_df = train_test_split(
        df,
        train_size=0.05,
        stratify=df['author']
    )

    train_loader = build_supervised_dataset(
        train_df,
        tokeniser=tokeniser,
        batch_size=GLOBAL_BATCH_SIZE,
        view_size=VIEW_SIZE,
        max_seq_len=MAX_LEN
    )

    # Initialise trainer
    print(f"Intialising trainer")
    train_model = ContrastiveTrainer(model,
                                    device=DEVICE,
                                    learning_rate=1e-4,
                                    weight_decay=None, 
                                    epochs=EPOCHS, 
                                    minibatch_size=MINIBATCH_SIZE
                                    )
    
    print("Starting training")
    train(train_model, train_loader, EPOCHS)