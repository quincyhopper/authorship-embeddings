import torch
import random
from datasets import load_dataset
import lightning as L
from torch.utils.data import Dataset, DataLoader

class AuthorshipDataset(Dataset):
    def __init__(self, dataset, view_size: int):

        self.dataset = dataset
        self.view_size = view_size

        # Only select author IDs to save memory
        df_idxs = dataset.select_columns(['author']).to_pandas().reset_index()
        self.grouped_idxs = df_idxs.groupby('author')['index'].apply(list).to_dict()

        # Create list of author IDs
        self.author_ids = list(self.grouped_idxs.keys())

    def __len__(self):
        return len(self.author_ids)
    
    def __getitem__(self, index):
        author_id = self.author_ids[index]
        row_idxs = self.grouped_idxs[author_id]
        
        # Sample row indices
        sampled_idxs = random.choices(row_idxs, k=self.view_size)

        # Fetch actual text data
        samples = [self.dataset[int(i)]['text'] for i in sampled_idxs]

        return {"label": index, "texts": samples}
    
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
    def __init__(self, data_path, tokenizer, batch_size=256, view_size=3, max_seq_len=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.view_size = view_size
        self.max_seq_len = max_seq_len

    def setup(self, stage=None):

        # Load parquet file
        data = load_dataset(path='parquet', data_files=self.data_path, split='train')

        split = data.train_test_split(test_size=0.2, train_size=0.8)
        self.train_ds = split['train']
        self.val_ds = split['test']

    def train_dataloader(self):
        dataset = AuthorshipDataset(self.train_ds, self.view_size)
        collator = AuthorshipCollator(self.tokenizer, self.view_size, self.max_seq_len)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=collator, num_workers=1, drop_last=True)
    
    def val_dataloader(self):
        dataset = AuthorshipDataset(self.val_ds, self.view_size)
        collator = AuthorshipCollator(self.tokenizer, self.view_size, self.max_seq_len)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=collator, num_workers=2, drop_last=True)
