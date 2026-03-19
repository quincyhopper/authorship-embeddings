import torch
import random
import lightning as L
from collections import Counter, defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class AuthorshipDataset(Dataset):
    def __init__(self, dataset: Dataset, view_size: int, author_list: list):

        self.dataset = dataset
        self.view_size = view_size
        self.author_list = author_list

        # Only select author IDs to save memory
        # Keys: author, Values: [chunk indices]
        self.author_chunk_idxs = defaultdict(list)
        for chunk_idx, author in enumerate(dataset['author']):
            self.author_chunk_idxs[author].append(chunk_idx)

    def __len__(self):
        return len(self.author_list)
    
    def __getitem__(self, index: int):
        """Called repeatedly by DataLoader to make a batch. 
        
        Args:
            index: index of an author in self.author_ids

        Returns:
            Dictionary where values are lists. Get passed to __call__ of AuthorshipCollator.
        """
        
        # Sample chunks indices from this author
        chunk_idxs = self.author_chunk_indices[index]
        sampled_idxs = random.choices(chunk_idxs, k=self.view_size)

        # Fetch input_ids
        input_ids = [self.dataset[int(i)]['input_ids'] for i in sampled_idxs] # [V, Seq_len]
        attention_mask = [self.dataset[int(i)]['attention_mask'] for i in sampled_idxs]

        return {"label": index, "input_ids": input_ids, 'attention_mask': attention_mask}
    
class AuthorshipCollator:
    def __int__(self, max_length, pad_token_id=1):
        self.max_length = max_length
        self.pad_token_id = pad_token_id # RoBERTa padding token ID is 1

    def __call__(self, batch: list[dict]):
        labels = torch.tensor([item['label'] for item in batch])

        # Pad tokens if necessary
        padded_input_ids = []
        padded_attention_masks = []
        for item in batch:
            ids = item['input_ids']
            mask = item['attention_mask']
            padding_needed = self.max_length - len(ids)
            
            # Pad inputs ids
            padded_ids = ids + [self.pad_token_id] * padding_needed
            padded_input_ids.append(padded_ids)

            # Pad mask
            padded_mask = mask + [0] * padding_needed
            padded_attention_masks.append(padded_mask)
            
        return (
            torch.tensor(padded_input_ids),
            torch.tensor(padded_attention_masks),
            labels
        )
    
class AuthorshipDataModule(L.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=1024, view_size=16, max_seq_len=512, num_workers=1):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.view_size = view_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Load train and val data
        self.train_raw = load_dataset(path='parquet', data_files=self.train_path, split='train')
        self.val_raw = load_dataset(path='parquet', data_files=self.val_path, split='train')

        # Extract author lists 
        train_authors = self.train_raw.unique('author')
        val_authors = self.val_raw.unique('author')

        # Init Dataset objects
        self.train_ds = AuthorshipDataset(self.train_raw, self.view_size, train_authors)
        self.val_ds = AuthorshipDataset(self.val_raw, self.view_size, val_authors)

        # Calculate weights for training sampler
        self.weights = self._calculate_weights(self.train_raw, train_authors)

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
            self.train_ds, 
            batch_size=self.batch_size, 
            sampler=sampler,
            collate_fn=AuthorshipCollator(), 
            num_workers=self.num_workers, 
            drop_last=True
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=AuthorshipCollator(), 
            )
