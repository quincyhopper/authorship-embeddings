import torch
import pandas as pd
import random
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class AuthorshipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, view_size: int):

        self.view_size = view_size

        # Group by authors -> {author_id: [text1, text2, ...]}
        self.grouped_data = df.groupby('author')['text'].apply(list).to_dict()

        # Create list of author IDs
        self.author_ids = list(self.grouped_data.keys())

    def __len__(self):
        return len(self.author_ids)
    
    def __getitem__(self, index):
        author_id = self.author_ids[index]
        all_texts = self.grouped_data[author_id] # List
        
        # Select V texts from author
        samples = random.choices(all_texts, k=self.view_size)

        return {
            "label": index,    # List index is unique integer ID
            "texts": samples # List of V strings
        }
    
class AuthorshipCollator:
    def __init__(self, tokenizer, view_size, max_len=512):
        self.tokenizer = tokenizer
        self.view_size = view_size
        self.max_len = max_len

    def __call__(self, batch):

        # Extract labels
        labels = torch.tensor([item['label'] for item in batch])
        
        # Flatten all texts: [Batch * Views]
        flat_texts = []
        for item in batch:
            flat_texts.extend(item['texts'])
            
        # Tokenize
        tokenized = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Reshape into the "cube": [Batch, Views, SeqLen]
        input_ids = tokenized['input_ids'].view(len(batch), self.view_size, -1)
        attention_mask = tokenized['attention_mask'].view(len(batch), self.view_size, -1)
        
        return input_ids, attention_mask, labels
    
class AuthorshipDataModule(L.LightningDataModule):
    def __init__(self, df, tokenizer, batch_size=256, view_size=3, max_seq_len=512):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.view_size = view_size
        self.max_seq_len = max_seq_len

    def setup(self, stage=None):
        self.train_df, self.val_df = train_test_split(
            self.df, train_size=0.8, stratify=self.df['author']
        )

    def train_dataloader(self):
        dataset = AuthorshipDataset(self.train_df, self.view_size)
        collator = AuthorshipCollator(self.tokenizer, self.view_size, self.max_seq_len)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=1,
            drop_last=True
        )
    
    def val_dataloader(self):
        dataset = AuthorshipDataset(self.val_df, self.view_size)
        collator = AuthorshipCollator(self.tokenizer, self.view_size, self.max_seq_len)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            drop_last=True
        )
