"""
This script is for counting all of the words across the 4 datasets. This dictionary will then be used in the data preparation stage: for each chunk, each word will have its rank, which can then be used in the masking stage.
"""
import json
from collections import Counter
from transformers import AutoTokenizer
from pathlib import Path
from datasets import load_dataset, Dataset, Features, Value, interleave_datasets

from prepare_twitter_reddit_blog import clean_twitter

def count_words(batch, tokenizer) -> dict:
    """Takes a batch of texts from ds.map and returns a dictionary of all the counts of words"""

    word_counts = Counter()
    encodings = tokenizer(batch['text'], add_special_tokens=False) # No need for special tokens (BOS, EOS)

    for i, text in enumerate(batch['text']):
        
        word_ids = encodings.word_ids(i)
        offsets = encodings.encodings[i].offsets # subword boundaries

        # Build a map of word_id -> word
        word_to_bounds = {}
        for token_idx, w_id in enumerate(word_ids):
            if w_id is not None:
                start, end = offsets[token_idx]
                if w_id not in word_to_bounds:
                    word_to_bounds[w_id] = [start, end]
                else:
                    # Increment end character index
                    word_to_bounds[w_id][-1] = end

        for w_id, (start, end) in word_to_bounds.items():
            word_str = text[start:end].strip().lower()
            word_counts[word_str] += 1
    
    # Return dictionary to work with Dataset.map 
    return {"counts": [dict(word_counts)]}

def count_words_in_dataset(ds: Dataset, tokenizer):
    global_word_counts = Counter()

    batch_counts = ds.map(
        count_words,
        batched=True,
        batch_size=5000,
        remove_columns=ds.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    for row in batch_counts:
        global_word_counts.update(row['counts'])

    return global_word_counts


if __name__ == "__main__":

    RAW_FEATURES = Features({
        "author": Value("string"),
        "text": Value("string"),
        "doc_id": Value("string"),
        "source": Value("string")
    })

    data_dir = Path(__file__).resolve().parent.parent
    
    base_files = [
        data_dir / 'blogtext_raw.parquet',
        data_dir / 'reddit_raw.parquet',
        data_dir / 'gutenberg_raw.parquet'
    ]

    twitter_files = [
        data_dir / 'twitter_train_raw.parquet',
        data_dir / 'twitter_test_raw.parquet'
    ]

    # Load separately so we can apply cleaning only to Twitter
    ds_base = load_dataset(path='parquet', data_files=base_files, split='train', features=RAW_FEATURES, streaming=True)
    ds_twitter = load_dataset(path='parquet', data_files=twitter_files, split='train', features=RAW_FEATURES, streaming=True)
    ds_twitter = ds_twitter.map(clean_twitter, batched=True, batch_size=10000)
    final_ds = interleave_datasets([ds_base, ds_twitter])

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ["<u>", "<h>"]}) # Add special tokens for Twitter
    
    word_counts = count_words_in_dataset(final_ds, tokenizer)
    with open('word_counts.json', 'w') as f:
        json.dump(word_counts, f)

