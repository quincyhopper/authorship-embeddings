import torch
import random
import lightning as L
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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
        """Used by DataLoader to know the indices to sample from."""
        return len(self.author_ids)
    
    def __getitem__(self, index):
        """Called repeatedly by DataLoader to fill up a batch. 
        
        Args:
            index: index of an author in self.author_ids
        """
        author_id = self.author_ids[index]
        chunk_idxs = self.grouped_idxs[author_id]
        
        # Sample chunks indices from this author
        sampled_idxs = random.choices(chunk_idxs, k=self.view_size)

        # Fetch input_ids
        input_ids = [self.dataset[int(i)]['input_ids'] for i in sampled_idxs] # [V, Seq_len]
        attention_mask = [self.dataset[int(i)]['attention_mask'] for i in sampled_idxs]

        return {"label": index, "input_ids": input_ids, 'attention_mask': attention_mask}
    
class AuthorshipCollator:
    def __call__(self, batch):
        labels = torch.tensor([item['label'] for item in batch])
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        
        return input_ids, attention_mask, labels
    
class AuthorshipDataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size=1024, view_size=16, max_seq_len=512, num_workers=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.view_size = view_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Load dataset
        full_ds = load_dataset(path='parquet', data_files=self.data_path, split='train')

        # Filter authors with less than 16 chunks
        author_counts = Counter(full_ds['author'])
        valid_authors = {auth for auth, count in author_counts.items() if count >= 16}
        filtered_ds = full_ds.filter(lambda x: x['author'] in valid_authors)

        # Create train/val split
        split_ds = filtered_ds.train_test_split(test_size=0.2, seed=42)
        self.train_ds_raw = split_ds['train']
        self.val_ds_raw = split_ds['test']
        
        # Wrap AuthorshipDataset class
        self.train_dataset = AuthorshipDataset(self.train_ds_raw)
        self.val_dataset = AuthorshipDataset(self.val_ds_raw)

        # Calculate weights for 1/14 batch requirement
        self.weights = self._calculate_weights(self.train_ds_raw)

    def _calculate_weights(self, dataset):
        
        # Count the number of datasets and total samples
        sources = dataset['sources']
        source_counts = Counter(sources)
        total_samples = len(sources)

        # Each source gets AT LEAST 1/14 
        target_min_prob = 1/14

        # Map source to weight
        source_to_weight = {}
        for s, count in source_counts.items():
            default_prob = count / total_samples
            upsampled_prob = max(default_prob, target_min_prob)
            source_to_weight[s] = upsampled_prob / count

        weights = [source_to_weight[s] for s in sources]

        return torch.DoubleTensor(weights)

    def train_dataloader(self):

        sampler = WeightedRandomSampler(self.weights, num_samples=len(self.weights), replacement=True)
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            sampler=sampler,
            collate_fn=AuthorshipCollator(), 
            num_workers=self.num_workers, 
            drop_last=True
            )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=AuthorshipCollator(), 
            )
