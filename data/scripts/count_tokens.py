"""
Count token ids across the *_chunks.parquet files while skipping pure spaces, punctuation, and special tokens. This is an alternative to counting words.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Parallelize at process level
import json
import numpy as np
import pyarrow.parquet as pq
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED
from tqdm import tqdm
from transformers import AutoTokenizer

_VALID_TOKEN_MASK = None

def init_worker():
    """Executed once per background process to build a static token safety filter."""
    global _VALID_TOKEN_MASK
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<u>", "<h>"]})
    
    vocab_size = len(tokenizer)
    _VALID_TOKEN_MASK = np.zeros(vocab_size, dtype=bool)
    
    special_ids = {
        tokenizer.bos_token_id,  # <s>
        tokenizer.eos_token_id,  # </s>
        tokenizer.pad_token_id,  # <pad>
        tokenizer.unk_token_id,  # <unk>
    }
    
    # Pre-classify every possible token ID in the vocabulary
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
            
        token_str = tokenizer.convert_ids_to_tokens(token_id) # (e.g., 'Ġthis' or 'Ġ.')
        clean_str = tokenizer.convert_tokens_to_string([token_str]).strip()
        
        # Exclude numbers, punctuation, spaces, etc
        # <u> and <h> are preserved.
        if clean_str and any(c.isalpha() for c in clean_str):
            _VALID_TOKEN_MASK[token_id] = True

def count_token_batch(batch: dict) -> np.ndarray:
    global _VALID_TOKEN_MASK
    flat_ids = np.hstack(batch['input_ids'])
    counts = np.bincount(flat_ids, minlength=len(_VALID_TOKEN_MASK)) # Count ids

    # Element-wise multiply counts by dict values
    return counts * _VALID_TOKEN_MASK

def producer(file_paths: list[Path], batch_size:int):
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_size, columns=['input_ids']):
            yield rb.to_pydict()

def main(file_paths: list, output_path: str='token_counts.json'):
    batch_size=  4096
    total_rows = 0
    total_batches = 0
    for path in file_paths:
        n_rows = pq.read_metadata(path).num_rows
        total_rows += n_rows
        total_batches = (n_rows + batch_size -1) // batch_size

    print(f"Processing {len(file_paths)} files containing {total_rows:,} combined rows", flush=True)

    n_workers = 4
    max_inflight = n_workers * 3  

    # Init tokeniser just to get the vocab size
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<u>", "<h>"]})
    global_token_counts = np.zeros(len(tokenizer), dtype=np.int64)

    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as ex:
        inflight = set()
        pbar = tqdm(total=total_batches, desc="Compiling Global Token ID Counts")
        for batch in producer(file_paths, batch_size):
            inflight.add(ex.submit(count_token_batch, batch))
            if len(inflight) >= max_inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    global_token_counts += fut.result()
                    pbar.update(1)
        for fut in as_completed(inflight):
            global_token_counts += fut.result()
            pbar.update(1)
        pbar.close()

    # Filter out IDs with zero occurrences
    output_counts_dict = {
        int(token_id): int(count)
        for token_id, count in enumerate(global_token_counts)
        if count > 0
    }

    data_dir = Path(__file__).resolve().parent.parent
    output_path = data_dir / output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_counts_dict, f, indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Count token id frequencies across 1 or more parquet files.")
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help="Space-separated paths to one or more parquet files that each have an input_ids column."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='token_counts.json',
        help='Output path of the json file (will save in data directory).'
    )
    args = parser.parse_args()

    file_paths = [Path(p).resolve() for p in args.inputs]
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(f"File path does not exist: {path}")
    
    main(args.inputs, args.output)