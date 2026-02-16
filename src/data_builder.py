import torch
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class AuthorshipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, view_size=16, ):

        self.view_size = view_size

        # Group by authors -> {author_id: [text1, text2, ...]}
        self.grouped_data = df.groupby('author')['text'].apply(list).to_dict()
        self.author_ids = list(self.grouped_data.keys())

    def __len__(self):
        return len(self.author_ids)
    
    def __getitem__(self, index):
        author_id = self.author_ids[index]
        all_texts = self.grouped_data[author_id] # List
        
        samples = random.choices(all_texts, k=self.view_size)

        return {
            "texts": samples, # List of view_size strings
            "label": index    # List index is unique integer ID
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
    
def build_supervised_dataset(df: pd.DataFrame, tokeniser, batch_size=1024, view_size=16, max_seq_len=512, max_docs=10000):
    """Tokenise texts and arange into batches.

    Args:
        df: Dataframe with columns 'author' and 'text'.
        tokeniser: instance of a tokeniser.
        batch_size: the number of unique authors in a single training step. For example, if you want to contrast 1,024 authors, batch_size should be 1,024.
        view_size: the number of different documents to sample from each author in a single batch. Must be greater than 1.
        max_seq_len: maximum number of tokens allowed per document.
        max_docs: maximum number of total documents to sample

    Returns:
        Dataloader containig batches of tokenised sequences of shape [B, V, L]
    """

    if max_docs and len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)

    dataset = AuthorshipDataset(df=df, view_size=view_size)
    collator = AuthorshipCollator(tokeniser, view_size, max_seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )