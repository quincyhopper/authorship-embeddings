import torch
import random
import lightning as L
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Sampler

def make_attn_mask(word_ranks, masking_threshold):
    mask = (word_ranks > masking_threshold)
    return mask.long()

class AuthorshipDataset(Dataset):
    def __init__(self, dataset, view_size: int, author_list: list, masking_threshold: int, is_validation: bool=False):

        self.dataset = dataset.with_format('torch', columns=['input_ids', 'word_ids', 'word_ranks'], output_all_columns=True)
        self.view_size = view_size
        self.author_list = author_list
        self.masking_threshold = masking_threshold
        self.is_validation = is_validation

        # Only select author IDs to save memory
        # Keys: author, Values: [chunk indices]
        self.author_chunk_idxs = defaultdict(list)
        for chunk_idx, author in enumerate(dataset['author']):
            self.author_chunk_idxs[author].append(chunk_idx)

    def __len__(self):
        return len(self.author_list)
    
    def __getitem__(self, index: int):
        """Called repeatedly by DataLoader to make a batch. If validation dataset, we also need to pass the split
        
        Args:
            index: index of an author in self.author_ids

        Returns:
            Dictionary where values are lists. Get passed to __call__ of AuthorshipCollator.
        """
        
        # Sample chunks indices from this author
        author = self.author_list[index]
        chunk_idxs = self.author_chunk_idxs[author]

        if self.is_validation:
            sampled_idxs = chunk_idxs[:self.view_size] # Deterministically get chunks
        else:
            sampled_idxs = random.sample(chunk_idxs, k=self.view_size)

        # Fetch input_ids
        input_ids = self.dataset[sampled_idxs]['input_ids'] # [V, Seq_len]
        word_ranks = self.dataset[sampled_idxs]['word_ranks']   # [V, Seq_len]
        attn_masks = make_attn_mask(word_ranks, self.masking_threshold)

        return {"label": index, "input_ids": input_ids, 'attention_mask': attn_masks}
    
class AuthorshipCollator:
    def __call__(self, batch: list[dict]):
        labels = [item['label'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return input_ids, attention_mask, labels
    
class BalancedSampler(Sampler):
    def __init__(self, author_list: list, author_source_map: dict, batch_size: int):

        self.batch_size = batch_size
        self.num_corpora = 4
        if batch_size % self.num_corpora != 0:
            raise ValueError(f"Batch size {self.batch_size} must be divisible by num corpora ({self.num_corpora})")
        
        self.samples_per_corpus = batch_size // self.num_corpora

        self.source_to_idxs = defaultdict(list)
        for idx, author in enumerate(author_list):
            source = author_source_map[author]
            self.source_to_idxs[source].append(idx)

        self.sources = list(self.source_to_idxs.keys())

    def __iter__(self):
        # Shuffle the indices of each source
        shuffled_indices = {
            src: random.sample(idxs, len(idxs))
            for src, idxs in self.source_to_idxs.items()
        }

        # Get length of largest corpus
        max_len = max([len(x) for x in shuffled_indices.values()])
        num_batches = (max_len + self.samples_per_corpus -1) // self.samples_per_corpus

        # Assemble batches
        for _ in range(num_batches):
            batch = []
            for source, pool in shuffled_indices.items():
                if len(pool) < self.samples_per_corpus: # Up sample if source runs out
                    shuffled_indices[source] = random.sample(self.source_to_idxs[source], len(self.source_to_idxs[source]))
                    pool = shuffled_indices[source]
                
                for _ in range(self.samples_per_corpus):
                    batch.append(pool.pop())

            random.shuffle(batch)
            yield(batch)

    def __len__(self):
        max_len = max([len(x) for x in self.source_to_idxs.values()])
        return (max_len + self.samples_per_corpus -1) // self.samples_per_corpus

class AuthorshipDataModule(L.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=1024, view_size=16, max_seq_len=512, num_workers=1, masking_threshold: int=0):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.view_size = view_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.masking_threshold = masking_threshold

    def setup(self, stage=None):

        # Load train and val data
        self.train_raw = load_dataset(path='parquet', data_files=self.train_path, split='train')
        self.val_raw = load_dataset(path='parquet', data_files=self.val_path, split='train')

        # Extract author lists 
        self.train_authors = self.train_raw.unique('author')
        val_authors = self.val_raw.unique('author')

        # Init Dataset objects
        self.train_ds = AuthorshipDataset(self.train_raw, self.view_size, self.train_authors, self.masking_threshold)
        self.val_ds = AuthorshipDataset(self.val_raw, self.view_size, val_authors, self.masking_threshold, is_validation=True)

        # Build author source map 
        self.author_source_map = {}
        for row in self.train_raw.select_columns(['author', 'source']):
            if row['author'] not in self.author_source_map:
                self.author_source_map[row['author']] = row['source']

    def train_dataloader(self):
        sampler = BalancedSampler(self.train_authors, self.author_source_map, self.batch_size)
        return DataLoader(
            self.train_ds, 
            batch_sampler=sampler,
            collate_fn=AuthorshipCollator(), 
            num_workers=self.num_workers, 
            #drop_last=True
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=AuthorshipCollator(), 
            )
