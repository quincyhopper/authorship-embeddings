import torch
import torch.distributed as dist
import random
import lightning as L
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Sampler

class AuthorshipDataset(Dataset):
    def __init__(self, dataset, view_size: int, author_list: list, is_validation: bool=False, content_masking: bool=False):
        """
        Args:
            dataset: training or validation dataset.
            view_size: number of texts per author in a batch.
            author_list: list of unique authors in the dataset.
            is_validation: if True, select the first view_size texts from each author. If False, randomly sample view_size texts from each author.
            content_masking: if True, __getitem__ returns the word ranks in addition to labels and input_ids. If False, it just returns labels and input_ids.
        """
        if content_masking:
            columns = ['input_ids', 'word_ranks']
        else:
            columns = ['input_ids']
        
        self.dataset = dataset.with_format('torch', columns=columns, output_all_columns=True)
        self.view_size = view_size
        self.author_list = author_list
        self.is_validation = is_validation
        self.content_masking = content_masking

        # Only select author IDs to save memory
        # Keys: author, Values: [chunk indices]
        self.author_chunk_idxs = defaultdict(list)
        for chunk_idx, author in enumerate(dataset['author']):
            self.author_chunk_idxs[author].append(chunk_idx)

    def __len__(self):
        return len(self.author_list)
    
    def __getitem__(self, index: int) -> dict:
        """Called repeatedly by DataLoader to make a batch. If validation dataset, we take the first view_size chunks from the author.
        
        Args:
            index: index of an author in self.author_ids

        Returns:
            Dictionary where values are lists. A batch of these get passed to AuthorshipCollator.__call__.
        """
        
        # Sample chunks indices from this author
        author = self.author_list[index]
        chunk_idxs = self.author_chunk_idxs[author]

        if self.is_validation:
            sampled_idxs = chunk_idxs[:self.view_size] # Deterministically get chunks
        else:
            sampled_idxs = random.sample(chunk_idxs, k=self.view_size)

        input_ids = self.dataset[sampled_idxs]['input_ids'] # [V, Seq_len]

        if self.content_masking:
            word_ranks = self.dataset[sampled_idxs]['word_ranks']   # [V, Seq_len]
            return {"label": index, "input_ids": input_ids, 'word_ranks': word_ranks}
        else:
            return {"label": index, "input_ids": input_ids}
    
class AuthorshipCollator:
    def __init__(self, content_masking: bool=False, masking_threshold: int | None=None):
        """
        Args:
            content_masking: if True, __call__ expects word ranks and uses them to apply a content mask. If False, it just expects labels and input_ids and does no masking.
            masking_threhold: words with a frequency <= to threshold will be replaced with <mask>.
        """
        if content_masking and masking_threshold is None:
            raise ValueError("content_masking set to True but masking_threshold is None.")

        self.content_masking = content_masking
        self.masking_threshold = masking_threshold
        self.mask_token_id = 50264 # roberta-large's masking token id

    def __call__(self, batch: list[dict]):
        """Take multiple outputs from AuthorshipDataset.__getitem__ and combine into one batch; create attention masks; optionally mask the input_ids,"""

        labels = torch.tensor([item['label'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch]) # Shape (batch_size, view_size, 512)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # chunks are always 512-tokens so padding is never present

        if self.content_masking:
            word_ranks = torch.stack([item['word_ranks'] for item in batch])
            mask = (word_ranks <= self.masking_threshold) & (word_ranks != -1)
            input_ids[mask] = self.mask_token_id

        return input_ids, attention_mask, labels
    
class BalancedSampler(Sampler):
    def __init__(self, author_list: list, author_source_map: dict, batch_size: int):
        
        # Detect distributed environment rank and world size
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Divide global batch size evenly among GPUs to get per-GPU batch size
        if batch_size % self.world_size != 0:
            raise ValueError(f"Global batch size {batch_size} must be divisible by world size {self.world_size}")

        self.local_batch_size = batch_size // self.world_size
        self.num_corpora = 4

        if self.local_batch_size % self.num_corpora != 0:
            raise ValueError(
                f"Per-GPU batch size {self.local_batch_size} must be divisible by the number of corpora ({self.num_corpora})")
        
        # NOTE: if anything other than 4 corpora are in the train set, this could cause weird batch sizes
        # This will be 256 when GLOBAL_BATCH_SIZE = 4096 on 4 GPUs
        self.samples_per_corpus = self.local_batch_size // self.num_corpora 

        # Map all sources to their indices
        full_source_to_idxs = defaultdict(list)
        for idx, author in enumerate(author_list):
            source = author_source_map[author]
            full_source_to_idxs[source].append(idx)

        # Shard each corpus list so this GPU only samples from a distinct slice
        self.source_to_idxs = defaultdict(list)
        for source, idxs in full_source_to_idxs.items():
            # Sorting guarantees consistent sharding order across all parallel GPU workers
            sorted_idxs = sorted(idxs)  
            self.source_to_idxs[source] = sorted_idxs[self.rank::self.world_size]

        self.sources = list(self.source_to_idxs.keys())
        self.epoch = 0

        self.sources = list(self.source_to_idxs.keys())

    def set_epoch(self, epoch: int):
        """Required by PyTorch Lightning"""
        self.epoch = epoch

    def __iter__(self):
        
        # Ensure shuffling changes between epochs but that across the GPUs, shufflers stay isolated
        seed = self.epoch + self.rank * 1000
        local_random = random.Random(seed)

        # Shuffle the indices of each source
        shuffled_indices = {
            src: local_random.sample(idxs, len(idxs))
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
    def __init__(self, train_path, val_path, batch_size=4096, view_size=16, max_seq_len=512, num_workers=1, content_masking: bool=False, masking_threshold: int | None=None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.view_size = view_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        if content_masking and masking_threshold is None:
            raise ValueError(f"content_masking set to True but masking_threshold is None.")
        
        self.content_masking = content_masking
        self.masking_threshold = masking_threshold

    def setup(self, stage=None):

        # Load train and val data
        self.train_raw = load_dataset(path='parquet', data_files=self.train_path, split='train')
        self.val_raw = load_dataset(path='parquet', data_files=self.val_path, split='train')

        # Extract author lists 
        self.train_authors = self.train_raw.unique('author')
        val_authors = self.val_raw.unique('author')

        # Init Dataset objects
        self.train_ds = AuthorshipDataset(self.train_raw, self.view_size, self.train_authors, content_masking=self.content_masking)
        self.val_ds = AuthorshipDataset(self.val_raw, self.view_size, val_authors, is_validation=True, content_masking=self.content_masking)

        # Build author source map 
        self.author_source_map = {}
        seen = set()
        for row in self.train_raw.select_columns(['author', 'source']):
            if row['author'] not in seen:
                self.author_source_map[row['author']] = row['source']
                seen.add(row['author'])
            if len(seen) == len(self.train_authors):
                break # Break early if we have seen all unique authors

    def train_dataloader(self):
        sampler = BalancedSampler(self.train_authors, self.author_source_map, self.batch_size)
        return DataLoader(
            self.train_ds, 
            batch_sampler=sampler,
            collate_fn=AuthorshipCollator(content_masking=self.content_masking, masking_threshold=self.masking_threshold), 
            num_workers=self.num_workers, 
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=AuthorshipCollator(content_masking=self.content_masking, masking_threshold=self.masking_threshold), 
            num_workers=self.num_workers,
            )
