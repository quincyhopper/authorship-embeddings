"""
Counts all of the words in the training dataset. Words are decoded from the input_ids using word_ids to calculate boundaries.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we parallelise at the process level
import json
import pyarrow.parquet as pq
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- per-worker globals (loaded once via the pool initializer) ----
_TOKENIZER = None
_ID_TO_TOKEN = None

def init_worker():
    global _TOKENIZER, _ID_TO_TOKEN
    tok = AutoTokenizer.from_pretrained("roberta-large")
    tok.add_special_tokens({"additional_special_tokens": ["<u>", "<h>"]})
    _TOKENIZER = tok
    _ID_TO_TOKEN = tok.convert_ids_to_tokens(list(range(len(tok))))

def count_batch(batch: dict):
    """Count the number of words in a batch of texts using the pretokeniser. This function is submitted to the queue using ex.submit(). 
    
    Args:
        batch: dictionary with keys input_ids and word_ids. Each value is a list of lists.
    """

    tok = _TOKENIZER
    id_to_token = _ID_TO_TOKEN
    counts = Counter()

    batch_input_ids = batch['input_ids']
    batch_word_ids = batch['word_ids']

    for input_ids, word_ids in zip(batch_input_ids, batch_word_ids):
        n = len(input_ids)
        start = 0
        while start < n:
            wid = word_ids[start]
            if wid == -1: # Skip special tokens
                start += 1
                continue

            end = start + 1
            while end < n and word_ids[end] == wid: # Increment until we span a whole word
                end += 1

            # Get all the tokens belonging to this word id
            tokens = [id_to_token[t] for t in input_ids[start:end]]
            word = tok.convert_tokens_to_string(tokens).strip().lower() # Reconstruct the original word

            if word and any(c.isalpha() for c in word):
                counts[word] += 1

            start = end # Move to next word id

    return counts

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    train_path = data_dir / 'train.parquet'

    batch_size = 2048
    n_rows = pq.read_metadata(train_path).num_rows
    total_batches = (n_rows + batch_size - 1) // batch_size
    print(f"Training set has {n_rows} rows", flush=True)

    def producer():
        """Generator to get batches of input_ids and word_ids"""
        pf = pq.ParquetFile(train_path)
        for rb in pf.iter_batches(batch_size=batch_size, columns=['input_ids', 'word_ids']):
            yield rb.to_pydict()

    n_workers = 4
    max_inflight = n_workers * 3  

    total = Counter()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as ex:
        inflight = set() # Unfinished batches
        pbar = tqdm(total=total_batches, desc="Compiling Global Word Counts")
        for batch in producer():
            inflight.add(ex.submit(count_batch, batch))
            if len(inflight) >= max_inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    total.update(fut.result())
                    pbar.update(1)
        for fut in as_completed(inflight):
            total.update(fut.result())
            pbar.update(1)
        pbar.close()

    output_path = str(data_dir / "word_counts.json")
    with open(output_path, "w") as f:
        json.dump(total, f)
    print(f"Wrote {len(total)} unique word types to {output_path}", flush=True)