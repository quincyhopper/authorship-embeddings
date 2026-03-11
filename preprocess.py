import re
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

NUM_PROC = 8

def create_train_val(data: list[str], train_size: float, rng: int=42):
    full_ds = load_dataset(path='parquet', data_files=DATA_PATH, split='train', features=raw_features, download_mode="force_redownload",)

    # Filter authors with less than 16 chunks
    author_counts = Counter(full_ds['author'])
    valid_authors = {auth for auth, count in author_counts.items() if count >= 16}
    filtered_ds = full_ds.filter(lambda x: x['author'] in valid_authors, num_proc=NUM_PROC)

    # Make a stratified train test split
    author_sources = filtered_ds.select_columns(['author', 'source']).to_pandas().drop_duplicates('author')
    authors = author_sources['author'].tolist()
    sources = author_sources['source'].tolist()
    train_authors, val_authors = train_test_split(
        authors,
        test_size=0.8,
        random_state=rng,
        stratify=sources
    )

    train_ds = full_ds.filter(lambda x: x['author'] in set(train_authors), num_proc=NUM_PROC)
    val_ds = full_ds.filter(lambda x: x['author'] in set(val_authors), num_proc=NUM_PROC)

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

    # Check if batch comes from Project Gutenberg (first and last chunks will be dropped)
    is_gutenberg = batch.get('source', [None])[0] == 'gutenberg'

    # Fill the batch
    for original_idx, chunk_indices in doc_to_chunk_indices.items():

        # Trim gutenberg docs with at least 3 chunks
        if is_gutenberg and len(chunk_indices) > 2:
            final_indices = chunk_indices[1:-1]
        else:
            final_indices = chunk_indices

        for idx in final_indices:
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
        print(f"Processing {source}")

        # Get rows from this dataset 
        ds = dataset.filter(lambda x: x['source'] == source, num_proc=NUM_PROC)

        # Apply cleaning if necessary
        conf = config.get(source)
        if conf and conf['cleaner']:
            ds = ds.map(conf['cleaner'], num_proc=NUM_PROC)

        # Apply packing if necessary (e.g. for Twitter)
        if conf and conf['pack']:
            ds = ds.map(pack_by_author, batched=True, batch_size=10000, num_proc=NUM_PROC)

        # Tokenise and chunk
        chunks = ds.map(
            tokenise_and_chunk,
            fn_kwargs={'tokenizer': tokenizer, 'chunk_size': chunk_size},
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names, 
            num_proc=NUM_PROC
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

    print("Processing val split")
    val_chunks = process_and_chunk(val_ds, CONFIG, tokenizer, chunk_size=512)

    train_chunks.to_parquet('data/train_chunks.parquet')
    val_chunks.to_parquet('data/val_chunks.parquet')

    print(f"Train contains {len(train_chunks)} rows")
    print(f"Train contains {len(train_chunks.unique('author'))} unique authors")

    print(f"Val contains {len(val_chunks)} rows")
    print(f"Train contains {len(val_chunks.unique('author'))} unique authors")