"""
This script is for counting all of the words across the 4 datasets. This dictionary will then be used in the data preparation stage: for each chunk, each word will have its rank, which can then be used in the masking stage.
"""
import json
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
from pathlib import Path
from datasets import load_dataset, Features, Value, interleave_datasets

def count_words_in_stream(streaming_ds, tokenizer, total_rows: int, batch_size=256):
    global_word_counts = Counter()
    
    # Calculate exact total iterations for tqdm based on the new batch size
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    # 1. Stream batches directly using .iter() to bypass ds.map generator overhead
    progress_bar = tqdm(
        streaming_ds.iter(batch_size=batch_size), 
        total=total_batches, 
        desc="Compiling Global Word Counts"
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        encodings = tokenizer(batch['text'], add_special_tokens=False)

        for i, text in enumerate(batch['text']):
            word_ids = encodings.word_ids(i)
            offsets = encodings.encodings[i].offsets 

            word_to_bounds = {}
            for token_idx, w_id in enumerate(word_ids):
                if w_id is not None:
                    start, end = offsets[token_idx]
                    if w_id not in word_to_bounds:
                        word_to_bounds[w_id] = [start, end]
                    else:
                        word_to_bounds[w_id][-1] = end

            for w_id, (start, end) in word_to_bounds.items():
                word_str = text[start:end].strip().lower()
                if word_str and any(char.isalpha() for char in word_str): # "3rd" gets counted but numbers by themselves don't
                    global_word_counts[word_str] += 1
                
        if batch_idx % 1000 == 0:
            gc.collect()

    return global_word_counts

def clean_twitter(batch):

    texts = pa.array(batch['text'])
    texts = pc.replace_substring_regex(texts, pattern=r'https?://\S+|www\.\S+', replacement='<h>')
    texts = pc.replace_substring_regex(texts, pattern=r'@\w+', replacement='<u>')

    batch['text'] = texts.to_pylist()
    return batch

if __name__ == "__main__":

    RAW_FEATURES = Features({
        "author": Value("string"),
        "text": Value("string"),
        "doc_id": Value("string"),
        "source": Value("string")
    })

    data_dir = Path(__file__).resolve().parent.parent
    
    base_files = [
        str(data_dir / 'blogtext_raw.parquet'),
        str(data_dir / 'reddit_raw.parquet'),
        str(data_dir / 'gutenberg_raw.parquet')
    ]

    twitter_files = [
        str(data_dir / 'twitter_train_raw.parquet'),
        str(data_dir / 'twitter_test_raw.parquet')
    ]

    total_rows = 0
    for file_path in base_files + twitter_files:
        parquet_meta = pq.read_metadata(file_path)
        total_rows += parquet_meta.num_rows
    print(f"Counted {total_rows} rows across all datasets", flush=True)

    # Load separately so we can apply cleaning only to Twitter
    print("Loading datasets...", flush=True)
    ds_base = load_dataset(path='parquet', data_files=base_files, split='train', features=RAW_FEATURES, streaming=True)
    ds_twitter = load_dataset(path='parquet', data_files=twitter_files, split='train', features=RAW_FEATURES, streaming=True)

    print("Cleaning Twitter", flush=True)
    ds_twitter = ds_twitter.map(clean_twitter, batched=True, batch_size=1000)

    print("Interleaving datasets", flush=True)
    final_ds = interleave_datasets([ds_base, ds_twitter], stopping_strategy='all_exhausted_without_replacement')

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ["<u>", "<h>"]}) # Add special tokens for Twitter
    
    word_counts = count_words_in_stream(final_ds, tokenizer, total_rows=total_rows, batch_size=256)
    output_path = str(data_dir / 'word_counts.json')
    with open(output_path, 'w') as f:
        json.dump(word_counts, f)

