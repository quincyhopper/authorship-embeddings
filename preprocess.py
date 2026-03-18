import pyarrow.compute as pc
import pyarrow as pa
import numpy as np 
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import (
    load_dataset, 
    disable_caching,
    Features,
    Value,
    Sequence,
    Dataset
    )

disable_caching()

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

NUM_PROC = 4

def clean_twitter(batch):

    texts = pa.array(batch['text'])
    texts = pc.replace_substring_regex(texts, pattern=r'https?://\S+|www\.\S+', replacement='<h>')
    texts = pc.replace_substring_regex(texts, pattern=r'@\w+', replacement='<u>')

    batch['text'] = texts.to_pylist()
    return batch

def preprocess(ds: Dataset, source_name, config):
    conf = config.get(source_name)

    if conf and conf['cleaner']:
        ds = ds.map(
            conf['cleaner'], 
            batched=True, 
            batch_size=1000, 
            num_proc=NUM_PROC, 
            desc=f"Cleaning {source_name}",
            load_from_cache_file=False
            )

    if conf and conf['pack'] and conf['sep']:
        df = ds.to_pandas()
        sep = conf['sep']
        
        if source_name == 'twitter':
            df = df.sample(frac=1).groupby('author').head(10000) # Sample 10,000 tweets per author
        
        # Join documents by author
        df_packed = df.groupby('author', as_index=False).agg({
            'text': lambda x: sep.join(x.astype(str)),
            'source': 'first',
            'doc_id': lambda x: f"packed_{x.iloc[0]}"
        })
        
        # Convert back to HF Dataset and clean up index
        ds = Dataset.from_pandas(df_packed)
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])

    return ds

def tokenise_and_chunk(examples, tokeniser, source_name:str, chunk_size=512):

    outputs = tokeniser(
        examples["text"],
        truncation=True,
        max_length=chunk_size,
        return_overflowing_tokens=True,
        stride=0
    )

    sample_map = outputs.pop("overflow_to_sample_mapping")
    new_batch = {k: [] for k in HF_FEATURES.keys()}
    chunks_by_sample = defaultdict(list)

    for i, original_idx in enumerate(sample_map):
        if len(outputs["input_ids"][i]) == chunk_size:
            chunk = {
                'author': examples['author'][original_idx],
                'doc_id': examples["doc_id"][original_idx],
                'source': examples['source'][original_idx],
                'input_ids': outputs['input_ids'][i],
                'attention_mask': outputs['attention_mask'][i],
            }
            chunks_by_sample[original_idx].append(chunk)

    # Edge removal for Gutenberg
    if source_name == 'gutenberg':
        for original_idx, chunks in chunks_by_sample.items():
            if len(chunks) > 2:
                chunks = chunks[1:-1]
            
            for chunk in chunks:
                for k in new_batch.keys():
                    new_batch[k].append(chunk[k])
            
    return new_batch

def process(ds: Dataset, source_name: str, tokeniser, chunk_size: int):
        return ds.map(
            tokenise_and_chunk,
            batched=True,
            batch_size=1000,
            fn_kwargs={'tokeniser': tokeniser, 'chunk_size': chunk_size, 'source_name': source_name},
            remove_columns=ds.column_names,
            num_proc=NUM_PROC,
            desc=f"Tokenising and chunking {source_name}",
            features=HF_FEATURES,
            load_from_cache_file=False
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

    train_author_set = set(train_author_ids)
    val_author_set = set(val_author_ids)

    train_ds = filtered_chunks.filter(
        lambda x:x['author'] in train_author_set,
        num_proc=NUM_PROC,
        desc="Filtering train authors"
    )

    val_ds = filtered_chunks.filter(
        lambda x:x['author'] in val_author_set,
        num_proc=NUM_PROC,
        desc="Filtering val authors"
    )

    return train_ds, val_ds 

def make_report(ds: Dataset, split: str):
    print(f"\nTotal {split} chunks: {len(ds)}")
    print(f"Total {split} unique authors: {len(ds.unique('author'))}")

if __name__ == "__main__":

    CONFIG = {
        'blog': {'cleaner': None, 'pack': True, 'sep': "\n\n\n"},
        'twitter': {'cleaner': clean_twitter, 'pack': True, 'sep': "\n\n\n"},
        'reddit': {'cleaner': None, 'pack': True, 'sep': "\n\n\n"},
        'gutenberg': {'cleaner': None, 'pack': True, 'sep': "\n\n\n"}
        }

    # Define filenames
    data_files = [
        'data/blogtext_raw.parquet',
    ]

    train_size = 1.0
    source_name = 'blog'
    train_output = 'data/blogtext_train.parquet'
    val_output = ''

    print('Loading dataset...')
    full_ds = load_dataset(path='parquet', data_files=data_files, split='train', features=RAW_FEATURES)
    clean_ds = preprocess(full_ds, source_name, CONFIG)

    tokeniser = AutoTokenizer.from_pretrained('roberta-large')
    chunks = process(clean_ds, source_name, tokeniser, chunk_size=512)

    filtered_chunks = filter_valid_authors(chunks)
    train_chunks, val_chunks = create_train_val(filtered_chunks, train_size=train_size, seed=42)

    train_chunks.to_parquet(train_output)
    make_report(train_chunks, 'train')

    if val_chunks is not None:
        val_chunks.to_parquet(val_output)
        make_report(val_chunks, 'val')