import re
import gc
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import (
    load_dataset, 
    concatenate_datasets,
    Features,
    Value,
    Sequence
    )

# Schema to make sure columns are read correctly
hf_features = Features({
    "author": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string"),
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    })

raw_features = Features({
    "author": Value("string"),
    "text": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string")
    })

DATA_PATH = [
    'data/gutenberg_raw.parquet',
    'data/blogtext_raw.parquet',
    'data/reddit_raw.parquet',
    'data/twitter_train_raw.parquet',
    'data/twitter_test_raw.parquet',
    ]

NUM_PROC = 6

def create_train_val(data: list[str], train_size: float, rng: int=42):

    # Load dataset(s)
    full_ds = load_dataset(
        path='parquet', 
        data_files=DATA_PATH, 
        split='train', 
        features=raw_features,
        )

    # Filter authors with less than 16 chunks
    author_counts = full_ds.select_columns(['author']).to_pandas()['author'].value_counts()
    valid_authors = set(author_counts[author_counts >= 16].index)
    filtered_ds = full_ds.filter(lambda x: x['author'] in valid_authors, num_proc=NUM_PROC, desc="Filtering valid authors")

    # Make a stratified train test split
    author_sources = filtered_ds.select_columns(['author', 'source']).to_pandas().drop_duplicates('author')
    authors = author_sources['author'].tolist()
    sources = author_sources['source'].tolist()

    if train_size >= 1.0:
        return filtered_ds.shuffle(seed=rng), None
    else:
        train_authors, val_authors = train_test_split(
            authors,
            train_size=train_size,
            random_state=rng,
            stratify=sources
        )

    # Clean up full ds
    del full_ds
    gc.collect()

    train_ds = full_ds.filter(lambda x: x['author'] in set(train_authors), num_proc=NUM_PROC, desc="Filtering for train authors")

    val_ds = None # Init as None in case train size is 100%
    if val_authors:
        val_ds = full_ds.filter(lambda x: x['author'] in set(val_authors), num_proc=NUM_PROC, desc="Filtering for val authors")

    return train_ds, val_ds

def tokenise_and_chunk(batch, tokenizer, chunk_size):

    # Tokenise the texts
    outputs = tokenizer(
        batch['text'],
        truncation=True,
        max_length=chunk_size,
        return_overflowing_tokens=True,
        stride=0
    )

    # Extract chunk indices and tokens
    sample_map = outputs.pop('overflow_to_sample_mapping') 
    input_ids = outputs['input_ids']
    attention_masks = outputs['attention_mask']

    # Group indices of full-sized chunks by their original document
    doc_to_chunk_indices = defaultdict(list)
    for chunk_idx, original_idx in enumerate(sample_map):
        if len(input_ids[chunk_idx]) == chunk_size:
            doc_to_chunk_indices[original_idx].append(chunk_idx)

    new_batch = {
        "author": [],
        "doc_id": [],
        "source": [],
        "input_ids": [],
        "attention_mask": []
    }

    # Fill the batch
    for original_idx, chunk_indices in doc_to_chunk_indices.items():
        for idx in chunk_indices:
            new_batch["author"].append(str(batch["author"][original_idx])) 
            new_batch["doc_id"].append(str(batch["doc_id"][original_idx])) 
            new_batch["source"].append(str(batch["source"][original_idx]))

            new_batch["input_ids"].append(input_ids[idx])
            new_batch["attention_mask"].append(attention_masks[idx])

    return new_batch

def pack_by_author(batch):
    """Group all texts in batch by author and join them with triple linespace."""

    author_map = defaultdict(list)
    for author, text in zip(batch['author'], batch['text']):
        author_map[author].append(text)

    packed_authors = []
    packed_texts = []
    packed_doc_ids = [] 

    for author, texts in author_map.items():
        packed_authors.append(author)
        packed_texts.append("\n\n\n".join(texts))
        packed_doc_ids.append(f"packed_{author}")

    return {
        'author': packed_authors,
        'text': packed_texts,
        'doc_id': packed_doc_ids,
        'source': batch['source'][:len(packed_authors)]
    }

def clean_twitter(example):
    text = example['text']
    text = re.sub(r'https?://\S+|www\.\S+', '<h>', text) # Hyperlinks
    text = re.sub(r'@\w+', '<u>', text) # Usernames
    example['text'] = text
    return example

def process_and_chunk(dataset, config, tokenizer, chunk_size):
    processed = []

    # Get the names of all of the sources
    sources = dataset.unique('source')

    for source in sources:

        # Get rows from this dataset 
        ds = dataset.filter(lambda x: x['source'] == source, num_proc=NUM_PROC, desc=f"Filtering {source} documents")

        # Apply cleaning if necessary
        conf = config.get(source)
        if conf and conf['cleaner']:
            ds = ds.map(conf['cleaner'], num_proc=NUM_PROC, desc=f"Cleaning {source}")

        # Apply packing if necessary (e.g. for Twitter)
        if conf and conf['pack']:
            ds = ds.map(pack_by_author, batched=True, batch_size=10000, num_proc=NUM_PROC, desc=f"Packing {source}")

        current_chunk_size = 1 if source == 'gutenberg' else 10000

        # Tokenise and chunk
        chunks = ds.map(
            tokenise_and_chunk,
            fn_kwargs={'tokenizer': tokenizer, 'chunk_size': chunk_size},
            batched=True,
            batch_size=current_chunk_size,
            remove_columns=ds.column_names, 
            num_proc=NUM_PROC,
            desc=f"Tokenising and chunking {source}"
            )
        
        # Cast schema
        chunks = chunks.cast(hf_features)

        processed.append(chunks)

    return concatenate_datasets(processed)

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

    train_ds, val_ds = create_train_val(DATA_PATH, train_size=0.8, rng=42)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    print("Processing training split")
    train_chunks = process_and_chunk(train_ds, CONFIG, tokenizer, chunk_size=512)
    train_chunks.to_parquet('data/train_chunks.parquet')

    if val_ds is not None:
        print("Processing val split")
        val_chunks = process_and_chunk(val_ds, CONFIG, tokenizer, chunk_size=512)
        val_chunks.to_parquet('data/val_chunks.parquet')    

    print("\n--- Train Set Statistics by Source ---")
    train_cols = train_chunks.select_columns(['author', 'source']).to_pandas()

    train_stats = train_cols.groupby('source').agg(
        num_chunks=('author', 'count'),
        num_unique_authors=('author', 'nunique')
    )
    print(train_stats)
    print(f"\nTotal Train Chunks: {len(train_chunks)}")
    print(f"Total Train Unique Authors: {len(train_chunks.unique('author'))}")

    if val_ds is not None:
        print("\n--- Val Set Statistics by Source ---")
        val_cols = val_chunks.select_columns(['author', 'source']).to_pandas()

        val_stats = val_cols.groupby('source').agg(
            num_chunks=('author', 'count'),
            num_unique_authors=('author', 'nunique')
        )
        print(val_stats)
        print(f"\nTotal Val Chunks: {len(val_chunks)}")
        print(f"Total Val Unique Authors: {len(val_chunks.unique('author'))}")
