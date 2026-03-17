import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import (
    load_dataset, 
    Features,
    Value,
    Sequence,
    Dataset
    )

# Schema to make sure columns are read correctly
HF_FEATURES = Features({
    "author": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string"),
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    })

RAW_FEATURES = Features({
    "author": Value("string"),
    "text": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string")
    })

NUM_PROC = 1

def filter_valid_authors(ds: Dataset):

    author_column = ds.data.column('author')
    value_counts = pc.value_counts(author_column)
    mask = pc.greater_equal(value_counts.field('counts'), 16)
    author_counts = value_counts.filter(mask)
    valid_authors = author_counts.field('values')
    full_mask = pc.is_in(ds.data.column('author'), value_set=valid_authors)
    indices = np.where(full_mask.to_numpy())[0]
    
    return ds.select(indices)

def create_train_val(filtered_ds: Dataset, train_size, seed):
    if train_size == 1.0:
        return filtered_ds.shuffle(seed), None
    
    authors = filtered_ds.unique('author')

    train_author_ids, val_author_ids = train_test_split(
        authors,
        train_size=train_size,
        random_state=seed,
    )

    current_authors = filtered_ds.data.column('author')
    train_idx = np.where(pc.is_in(current_authors, value_set=pa.array(train_author_ids)).to_numpy())[0]
    train_ds = filtered_ds.select(train_idx)

    val_idx = np.where(pc.is_in(current_authors, value_set=pa.array(val_author_ids)).to_numpy())[0]
    val_ds = filtered_ds.select(val_idx)

    return train_ds, val_ds 

def pack_by_author(batch):
    df = pd.DataFrame(batch)
    
    # Group by author and join their texts with triple linespace
    packed = df.groupby('author', as_index=False).agg({
        'text': lambda x: "\n\n\n".join(x),
        'source': 'first',
        'doc_id': lambda x: f"packed_{x.iloc[0]}"
    })
    
    return packed.to_dict('list')

def clean_twitter(batch):
    texts = batch['text']
    texts = pc.replace_matches(texts, r'https?://\S+|www\.\S+', '<h>') # Hyperlinks
    texts = pc.replace_matches(texts, r'@\w+', '<u>') # Usernames
    batch['text'] = texts
    return batch

def preprocess(ds, source_name, config):
     
    conf = config.get(source_name)
    if conf and conf['cleaner']:
        print(f"Cleaning {source_name}")
        ds = ds.map(conf['cleaner'], num_proc=NUM_PROC, desc=f"Cleaning {source_name}")

    # Apply packing if necessary (e.g. for Twitter)
    if conf and conf['pack']:
        print(f"Packing {source_name}")
        ds = ds.map(pack_by_author, batched=True, batch_size=10000, num_proc=NUM_PROC, desc=f"Packing {source_name}")

    return ds

def tokenise_and_chunk(examples, tokeniser, chunk_size=512):

    outputs = tokeniser(
        examples["text"],
        truncation=True,
        max_length=chunk_size,
        return_overflowing_tokens=True,
        stride=0
    )

    sample_map = outputs.pop("overflow_to_sample_mapping")
    new_batch = {k: [] for k in HF_FEATURES.keys()}

    for i, original_idx in enumerate(sample_map):
        # Only keep full-sized chunks to maintain consistency
        if len(outputs["input_ids"][i]) == chunk_size:
            new_batch["input_ids"].append(outputs["input_ids"][i])
            new_batch["attention_mask"].append(outputs["attention_mask"][i])
            new_batch["author"].append(examples["author"][original_idx])
            new_batch["doc_id"].append(examples["doc_id"][original_idx])
            new_batch["source"].append(examples["source"][original_idx])
            
    return new_batch

def process(dataset: Dataset, source_name: str, tokeniser, chunk_size: int):
        print(f"Tokenising and chunking {source_name}")

        return dataset.map(
            tokenise_and_chunk,
            batched=True,
            batch_size=1,
            fn_kwargs={'tokeniser': tokeniser, 'chunk_size': chunk_size},
            remove_columns=dataset.column_names,
            num_proc=NUM_PROC,
            desc="Tokenising and chunking"
        ).cast(HF_FEATURES)

def make_report(ds: Dataset, split: str):
    print(f"\nTotal {split} chunks: {len(ds)}")
    print(f"Total {split} unique authors: {len(train_chunks.unique('author'))}")

if __name__ == "__main__":

    CONFIG = {
        'blog': {
            'cleaner': None,
            'pack': False
        },
        'twitter': {
            'cleaner': clean_twitter,
            'pack': True
        },
        'reddit': {
            'cleaner': None,
            'pack': False
        },
        'gutenberg': {
            'cleaner': None,
            'pack': False
        }
    }

    # Define filenames
    data_files = [
        'data/gutenberg_raw.parquet',
    ]

    source_name = 'gutenberg'
    train_output = 'data/gutenberg_train.parquet'
    val_output = ''

    # Load dataset
    print('Loading dataset')
    full_ds = load_dataset(path='parquet', data_files=data_files, split='train')

    # Filter for authors with more than 16 texts
    print('Filtering authors')
    filtered_ds = filter_valid_authors(full_ds)

    # Get train/val splits
    print('Making train/val split')
    train_ds, val_ds = create_train_val(filtered_ds, train_size=1.0, seed=42)

    # Preprocess
    train_clean = preprocess(train_ds, source_name, CONFIG)

    # Load tokeniser
    print('Loading tokeniser')
    tokeniser = AutoTokenizer.from_pretrained('roberta-large')

    # Tokenise and chunk
    train_chunks = process(train_clean, source_name, tokeniser, chunk_size=512)
    train_chunks.to_parquet(train_output)

    # Do the same for val if necessary
    if val_ds is not None:
        val_clean = preprocess(val_ds, source_name, CONFIG)
        val_chunks = process(val_clean, source_name, tokeniser, chunk_size=512)
        val_chunks.to_parquet(val_output)

    make_report(train_chunks, 'train')
    if val_ds is not None:
        make_report(val_chunks, 'val')
