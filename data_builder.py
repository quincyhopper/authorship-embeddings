import torch
import torch.nn as nn
import pandas as pd
import random
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
    def __init__(self, tokenizer, view_size, max_len):
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
    
def filter_authors(df: pd.DataFrame, min_docs: int):

    author_counts = df['author'].value_counts()
    valid_authors = author_counts[author_counts >= min_docs].index

    return df[df['author'].isin(valid_authors)].copy()
    
def build_supervised_dataset(df: pd.DataFrame, tokeniser, batch_size=1024, view_size=16, max_seq_len=512):
    """Tokenise texts and arange into batches.

    Args:
        df: Dataframe with columns 'author' and 'text'.
        tokeniser: instance of a tokeniser.
        batch_size: the number of unique authors in a single training step. For example, if you want to contrast 1,024 authors, batch_size should be 1,024.
        view_size: the number of different documents to sample from each author in a single batch. Must be greater than 1.
        max_seq_len: maximum number of tokens allowed per document.

    Returns:
        Dataloader containig batches of tokenised sequences of shape [B, V, L]
    """

    dataset = AuthorshipDataset(df=df, view_size=view_size)
    collator = AuthorshipCollator(tokeniser, view_size, max_seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        drop_last=True
    )