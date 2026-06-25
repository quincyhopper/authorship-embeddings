"""
Tokenise and chunk the datasets into 512-token chunks for downstream fine-tuning.

Each chunk (row) stores:
    - input_ids: the 512 token ids the model trains on.
    - word_ids: a chunk-local word index (0,0,1,2,2,...) with -1 for special tokens. This records which tokens constitute a word.

Word ids are relative to the chunk simply so the integers will be smaller, and then we can use int16.

This stage does not depend on word_counts.json (the output of word_counts.py). Word ranks are derived later on using ranks.py. Furthermore, the original word strings are completely recoverable from input_ids, and word_ids supplies the word boundaries, so word ranks can be computed any number of times without having to run this expensive script.

To add any new preprocessing (e.g. sampling, removing certain authors), modify the pack_authors function.
"""

import gc
import pyarrow.compute as pc
import pyarrow as pa
import duckdb
import numpy as np
import os
import time
import argparse
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Features, Value, Sequence, Dataset
from transformers import AutoTokenizer

RAW_FEATURES = Features({
    "author": Value("string"),
    "text": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string")
    })

CHUNKED_FEATURES = Features({
    "author": Value("string"),
    "doc_id": Value("string"),
    "source": Value("string"),
    "input_ids": Sequence(Value("uint16")),
    "word_ids": Sequence(Value("int16")),
    })

NUM_PROC = 4
SEED = "42"

def clean_twitter(batch):
    """Anonymise usernames and URLs in Twitter rows."""
    texts = pa.array(batch['text'])
    texts = pc.replace_substring_regex(texts, pattern=r'https?://\S+|www\.\S+', replacement='<h>')
    texts = pc.replace_substring_regex(texts, pattern=r'@\w+', replacement='<u>')

    batch['text'] = texts.to_pylist()
    return batch

def pack_authors(source_name: str, settings: dict, data_dir: Path) -> Path:
    """Uses DuckDB to group, sample, and merge texts.
    
    Specifically:
        Twitter: aggregate texts, sample up to 10,000 docs per author.

        Gutenberg: do NOT aggregate texts, sample n authors and all of the texts.

        Reddit and Blog: aggregate texts, remove [deleted] and 'nan' authors.
    """
    output_tmp_path = data_dir / f"{source_name}_packed_tmp.parquet"
 
    input_files_str = ", ".join([f"'{str(data_dir / f)}'" for f in settings['files']])
    sep = settings['sep']
 
    print(f"    Executing author packing via DuckDB for {source_name}...")
 
    con = duckdb.connect()
 
    if source_name == 'twitter':
        con.execute(f"""
            COPY (
                WITH ranked_tweets AS (
                    SELECT author, text, doc_id, source,
                           row_number() OVER (PARTITION BY author ORDER BY hash(doc_id || '{SEED}')) as rn
                    FROM read_parquet([{input_files_str}])
                    WHERE author IS NOT NULL AND author NOT IN ('None', 'nan')
                )
                SELECT author,
                       string_agg(trim(text), '{sep}') as text,
                       'packed_' || min(doc_id) as doc_id,
                       'twitter' as source
                FROM ranked_tweets
                WHERE rn <= 10000
                GROUP BY author
            ) TO '{str(output_tmp_path)}' (FORMAT parquet, COMPRESSION snappy);
        """)
    elif source_name == 'gutenberg':
        # We do not aggergate texts here because we need to be able to remove the first and last chunk
        # of each book. 
        con.execute(f"""
            COPY (
                SELECT author,
                       trim(text) as text,
                       doc_id,
                       'gutenberg' as source
                FROM read_parquet([{input_files_str}])
                WHERE author IS NOT NULL
                  AND CAST(author AS VARCHAR) NOT IN ('[deleted]', 'None', 'nan')
            ) TO '{str(output_tmp_path)}' (FORMAT parquet, COMPRESSION snappy);
        """)
    else: # Blogtext and Reddit: just aggregate and remove unwanted authors
        con.execute(f"""
            COPY (
                SELECT author,
                       string_agg(trim(text), '{sep}') as text,
                       'packed_' || min(doc_id) as doc_id,
                       '{source_name}' as source
                FROM read_parquet([{input_files_str}])
                WHERE author IS NOT NULL 
                    AND CAST(author AS VARCHAR) NOT IN ('[deleted]', 'None', 'nan')
                GROUP BY author
            ) TO '{str(output_tmp_path)}' (FORMAT parquet, COMPRESSION snappy);
        """)
 
    con.close()
    return output_tmp_path

def is_first_or_last_chunk(chunk_idx: int, doc_idx: int, doc_chunk_map: dict) -> bool:
    """True if chunk is the first or last in its document (or doc has < 2 chunks). Used for filtering gutenberg."""
    chunk_indices = doc_chunk_map[doc_idx]
    return len(chunk_indices) < 2 or chunk_idx == chunk_indices[0] or chunk_idx == chunk_indices[-1]

def to_local_word_ids(word_ids):
    """Renumber word_ids to a chunk-local sequence (0,0,1,2,...). Special tokens (HF word_id None) -> -1."""
    out = []
    cur = -1
    prev = object()  # sentinel distinct from any id and from None
    for wid in word_ids:
        if wid is None:
            out.append(-1)
            prev = None
        else:
            if wid != prev:
                cur += 1
            out.append(cur)
            prev = wid
    return out

def chunk_batch(batch: dict, tokenizer, source_name: str):
    """Tokenise to 512-token chunks and attach a per-token chunk-local word index."""
    outputs = tokenizer(
        batch['text'],
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        stride=0,
    )
 
    sample_map = outputs.pop("overflow_to_sample_mapping")
    new_batch = {k: [] for k in CHUNKED_FEATURES.keys()}
 
    # Only needed for the Gutenberg first/last-chunk filter.
    doc_chunk_map = None
    if source_name == 'gutenberg':
        doc_chunk_map = defaultdict(list)
        for chunk_idx, doc_idx in enumerate(sample_map):
            doc_chunk_map[doc_idx].append(chunk_idx)
 
    for chunk_idx, enc in enumerate(outputs.encodings):
        input_ids = enc.ids
        if len(input_ids) != 512:
            continue

        doc_idx = sample_map[chunk_idx]
 
        if source_name == 'gutenberg' and is_first_or_last_chunk(chunk_idx, doc_idx, doc_chunk_map):
            continue
 
        word_ids = to_local_word_ids(enc.word_ids)
 
        new_batch['author'].append(batch['author'][doc_idx])
        new_batch['doc_id'].append(batch['doc_id'][doc_idx])
        new_batch['source'].append(batch['source'][doc_idx])
        new_batch['input_ids'].append(input_ids)
        new_batch['word_ids'].append(word_ids)
 
    return new_batch

def filter_valid_authors(ds: Dataset, n: int=16):
    """Drop authors with < n 512-token chunks."""
    print("Filtering authors")

    author_column = ds.data.column('author')
    value_counts = pc.value_counts(author_column)
    mask = pc.greater_equal(value_counts.field('counts'), n)
    author_counts = value_counts.filter(mask)
    valid_authors = author_counts.field('values')
    full_mask = pc.is_in(ds.data.column('author'), value_set=valid_authors)
    indices = np.where(full_mask.to_numpy())[0]

    return ds.select(indices)

CONFIG = {
    'blog':      {'pack': True, 'sep': " </s> <s> ", 'files': ["blogtext_raw.parquet"], 'batch_size': 256},
    'twitter':   {'pack': True, 'sep': "\n\n\n",      'files': ["twitter_train_raw.parquet", "twitter_test_raw.parquet"], 'batch_size': 256},
    'reddit':    {'pack': True, 'sep': " </s> <s> ", 'files': ["reddit_raw.parquet"], 'batch_size': 256},
    'gutenberg': {'pack': True, 'sep': " </s> <s> ", 'files': ["gutenberg_raw.parquet"], 'batch_size': 4},
}

def chunk_datasets(tokenizer, unify: bool=True, remove_tmp: bool=True):
    """Main function for tokenising and chunking the 4 datasets.
    
    Args:
        tokenizer: roberta-large tokenizer.
        unify (bool): if True, the final chunks are saved into one dataset called 'chunks.parquet'. If False, datasets are saved to their own files, e.g. 'gutenberg_chunks.parquet'.
        remove_tmp (bool): if True, the temporary files created by pack_authors() are deleted.
    """
    print(f"Unify={unify}")
    print(f"Remove_tmp={remove_tmp}")

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    processed_chunks = [] if unify else None
    temporary_files = []

    for source, settings in CONFIG.items():
        print(f"\n=== Processing Corpus Source: {source.upper()} ===")
 
        packed_parquet_path = pack_authors(source, settings, data_dir)
        temporary_files.append(packed_parquet_path)
 
        ds = load_dataset(path='parquet', data_files=[str(packed_parquet_path)],
                          split='train', features=RAW_FEATURES)
 
        if source == 'twitter':
            ds = ds.map(clean_twitter, batched=True, batch_size=5000,
                        desc="Applying Twitter text regular expressions")
            
        chunked_ds = ds.map(
            chunk_batch,
            batched=True,
            batch_size=settings['batch_size'],
            num_proc=NUM_PROC,
            fn_kwargs={'tokenizer': tokenizer, 'source_name': source},
            remove_columns=ds.column_names,
            features=CHUNKED_FEATURES,
            desc=f"Tokenising and chunking {source}",
        )

        if unify:
            processed_chunks.append(chunked_ds)
            del chunked_ds; gc.collect()
        else:
            print(f"Filtering {source} chunks")
            filtered = filter_valid_authors(chunked_ds, 16)

            print(f"Saving {source} chunks to parquet")
            filtered.to_parquet(str(data_dir / f'{source}_chunks.parquet'))
            del filtered; gc.collect()

    if unify:
        print("\nConsolidating all processed datasets into a master dataset...")
        master_dataset = concatenate_datasets(processed_chunks)

        print("Filtering master dataset")
        master_dataset = filter_valid_authors(master_dataset, 16)
    
        print("\nSaving finalized chunks to disk...")
        master_dataset.to_parquet(str(data_dir / 'chunks.parquet'))

    if remove_tmp:
        print("Cleaning up intermediate data fragments...")
        for temp_file in temporary_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    print(f"chunk_datasets.py started at: {time.ctime()}")

    p = argparse.ArgumentParser()
    p.add_argument("--unify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rm-tmp", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
 
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ["<u>", "<h>"]})
 
    chunk_datasets(tokenizer, unify=args.unify, remove_tmp=args.rm_tmp)

    print(f"chunk_datasets.py finished at: {time.ctime()}")