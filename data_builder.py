import torch
import random
import lightning as L
from collections import Counter, defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class AuthorshipDataset(Dataset):
    def __init__(self, dataset, view_size: int, author_list: list):

        self.dataset = dataset
        self.view_size = view_size
        self.author_ids = author_list

        # Only select author IDs to save memory
        self.grouped_idxs = defaultdict(list)
        for idx, author in enumerate(dataset['author']):
            self.grouped_idxs[author].append(idx)

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
        # Load all datasets
        full_ds = load_dataset(path='parquet', data_files=self.data_path, split='train')

        # Filter authors with less than 16 chunks
        author_counts = Counter(full_ds['author'])
        valid_authors = [auth for auth, count in author_counts.items() if count >= 16] # List so we can shuffle below

        # Create train/val split
        # Splitting by authors, not by chunks. This ensures that we never test on an author we train on
        random.seed(42)
        random.shuffle(valid_authors)
        split_idx = int(len(valid_authors) * 0.8)
        train_author_list = valid_authors[:split_idx]
        val_author_list = valid_authors[split_idx:]
        train_set = set(train_author_list)
        val_set = set(val_author_list)
        self.train_ds_raw = full_ds.filter(lambda x: x['author'] in train_set, num_proc=10)
        self.val_ds_raw = full_ds.filter(lambda x: x['author'] in val_set, num_proc=10)
        self.train_dataset = AuthorshipDataset(self.train_ds_raw, self.view_size, train_author_list)
        self.val_dataset = AuthorshipDataset(self.val_ds_raw, self.view_size, val_author_list)

        # Calculate weights for 1/14 batch requirement
        self.weights = self._calculate_weights(self.train_ds_raw, train_author_list)

    def _calculate_weights(self, dataset, author_list):
        """Calculate the weight to upsample the smaller datasets. Replicating the 1/14 minimum."""

        # Map each author to their source dataset {author: source}
        author_source_map = {}
        for row in dataset.select_columns(['author', 'source']):
            if row['author'] not in author_source_map:
                author_source_map[row['author']] = row['source']
        
        # Among the unique authors, count the number of times each source appears
        author_sources = [author_source_map[auth] for auth in author_list]
        source_counts = Counter(author_sources)
        total_authors = len(author_sources)

        # Each source gets AT LEAST 1/14 
        target_min_prob = 1/14
        source_weight_map = {}

        for s, count in source_counts.items():
            default_prob = count / total_authors
            prob = max(default_prob, target_min_prob) # Weight is AT LEAST 1/14
            source_weight_map[s] = prob / count

        # Generate one weight per author based on their source
        weights = [source_weight_map[author_source_map[auth]] for auth in author_list]
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
