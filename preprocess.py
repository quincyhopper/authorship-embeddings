import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import numpy as np 
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import (
    load_dataset, 
    Features,
    Value,
    Sequence,
    Dataset
    )

RAW_FEATURES = Features({
    "author": Value("string"),
    "text": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string")
    })

HF_FEATURES = Features({
    "author": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string"),
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    })

NUM_PROC = 1

def clean_twitter(batch):

    texts = pa.array(batch['text'])
    texts = pc.replace_substring_regex(texts, pattern=r'https?://\S+|www\.\S+', replacement='<h>')
    texts = pc.replace_substring_regex(texts, pattern=r'@\w+', replacement='<u>')

    batch['text'] = texts.to_pylist()
    return batch

def preprocess(ds: Dataset, source_name, config):
    conf = config.get(source_name)

    if conf and conf['cleaner']:
        ds = ds.map(conf['cleaner'], batched=True, batch_size=1000, num_proc=NUM_PROC, desc=f"Cleaning {source_name}")

    if conf and conf['pack']:
        df = ds.to_pandas()

        # Sample 10,000 tweets per author (using frac=1 in case author has <10,000 tweets)
        df = df.sample(frac=1).groupby('author').head(10000)
        
        # Join tweets with 3 newlines
        df_packed = df.groupby('author', as_index=False).agg({
            'text': lambda x: "\n\n\n".join(x.astype(str)),
            'source': 'first',
            'doc_id': lambda x: f"packed_{x.iloc[0]}"
        })
        
        # Convert back to HF Dataset and clean up index
        ds = Dataset.from_pandas(df_packed)
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])

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

def process(ds: Dataset, source_name: str, tokeniser, chunk_size: int):
        return ds.map(
            tokenise_and_chunk,
            batched=True,
            batch_size=1000,
            fn_kwargs={'tokeniser': tokeniser, 'chunk_size': chunk_size},
            remove_columns=ds.column_names,
            num_proc=NUM_PROC,
            desc=f"Tokenising and chunking {source_name}",
            features=HF_FEATURES
        )

def filter_valid_authors(ds: Dataset):
    print("Filtering authors")

    author_column = ds.data.column('author')
    value_counts = pc.value_counts(author_column)
    mask = pc.greater_equal(value_counts.field('counts'), 16)
    author_counts = value_counts.filter(mask)
    valid_authors = author_counts.field('values')
    full_mask = pc.is_in(ds.data.column('author'), value_set=valid_authors)
    indices = np.where(full_mask.to_numpy())[0]
    
    return ds.select(indices)

def create_train_val(filtered_chunks: Dataset, train_size, seed):
    if train_size == 1.0:
        print(f"Only creating train split (train_size={train_size})")
        return filtered_chunks.shuffle(seed), None
    
    print(f"Creating train/val split (train_size={train_size})")
    
    authors = filtered_chunks.unique('author')

    train_author_ids, val_author_ids = train_test_split(
        authors,
        train_size=train_size,
        random_state=seed,
    )

    current_authors = filtered_chunks.data.column('author')
    train_idx = np.where(pc.is_in(current_authors, value_set=pa.array(train_author_ids)).to_numpy())[0]
    train_ds = filtered_chunks.select(train_idx)

    val_idx = np.where(pc.is_in(current_authors, value_set=pa.array(val_author_ids)).to_numpy())[0]
    val_ds = filtered_chunks.select(val_idx)

    return train_ds, val_ds 

def make_report(ds: Dataset, split: str):
    print(f"\nTotal {split} chunks: {len(ds)}")
    print(f"Total {split} unique authors: {len(train_chunks.unique('author'))}")

if __name__ == "__main__":

    CONFIG = {
        'blog': {'cleaner': None, 'pack': False},
        'twitter': {'cleaner': clean_twitter, 'pack': True},
        'reddit': {'cleaner': None, 'pack': False},
        'gutenberg': {'cleaner': None, 'pack': False}
        }

    # Define filenames
    data_files = [
        'data/blogtext_raw.parquet',
    ]

    train_size = 1.0
    source_name = 'blogs'
    train_output = 'data/blogtext_train.parquet'
    val_output = ''

    print('Loading dataset...')
    full_ds = load_dataset(path='parquet', data_files=data_files, split='train', features=RAW_FEATURES)
    clean_ds = preprocess(full_ds, source_name, CONFIG)
    tokeniser = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    chunks = process(clean_ds, source_name, tokeniser, chunk_size=512)
    filtered_chunks = filter_valid_authors(chunks)
    train_chunks, val_chunks = create_train_val(filtered_chunks, train_size=train_size, seed=42)

    train_chunks.to_parquet(train_output)
    make_report(train_chunks, 'train')

    if val_chunks is not None:
        val_chunks.to_parquet(val_output)
        make_report(val_chunks, 'val')