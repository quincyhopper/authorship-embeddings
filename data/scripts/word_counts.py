"""
Counts all words across the 4 datasets, producing a {word: frequency} dictionary for the downstream rank-based masking stage.

Word boundaries are defined by the RoBERTa-large pretokeniser itself, so that it is completely compatibile with downstream tokenisation. Multiprocessing is used to make the script faster. The seff report for a successful run on a himem node is as follows:

Job ID: 15903926
Cluster: ###
User/Group: ###
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 04:59:07
CPU Efficiency: 97.10% of 05:08:03 core-walltime
Job Wall-clock time: 05:08:03
Memory Utilized: 16.15 GB
Memory Efficiency: 51.36% of 31.45 GB (31.45 GB/core)
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we parallelise at the process level
import json
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED
from tqdm import tqdm
from transformers import AutoTokenizer

from chunk_datasets import pack_authors_to_disk, CONFIG 

SPECIALS = ("<u>", "<h>")

# ---- per-worker globals (loaded once via the pool initializer) ----
_PRE = None
_NORM = None


def init_worker():
    global _PRE, _NORM
    tok = AutoTokenizer.from_pretrained("roberta-large")
    bt = tok.backend_tokenizer
    _PRE = bt.pre_tokenizer
    _NORM = bt.normalizer


def clean_twitter_list(texts):
    """Replace hyperlinks and usernames with special characters."""
    arr = pa.array(texts, type=pa.string())
    arr = pc.replace_substring_regex(arr, pattern=r"https?://\S+|www\.\S+", replacement="<h>")
    arr = pc.replace_substring_regex(arr, pattern=r"@\w+", replacement="<u>")
    return arr.to_pylist()


def count_batch(args):
    """Count the number of words in a batch of texts using the pretokeniser. This function is submitted to the queue using ex.submit(). 
    
    Args:
        args: output of producer(), which is a tuple of (list, bool).
    """
    texts, is_twitter = args
    if is_twitter:
        texts = clean_twitter_list(texts)

    counts = Counter()
    pre = _PRE
    norm = _NORM

    for text in texts:
        if not text:
            continue

        # Honour <u>/<h> as single special tokens (matches add_special_tokens downstream).
        for sp in SPECIALS:
            n = text.count(sp)
            if n:
                counts[sp] += n
                text = text.replace(sp, " ")

        if norm is not None:
            text = norm.normalize_str(text)

        # pre_tokenize_str -> [(token_str, (start, end)), ...]
        counts.update(
            w for (_t, (s, e)) in pre.pre_tokenize_str(text)
            if (w := text[s:e].strip().lower()) and any(c.isalpha() for c in w)
        )

    return counts

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent

    # Pack authors using the same filtering methods as chunk_datasets.py
    print("Packing the datasets")
    packed_paths = dict()
    for source, settings in CONFIG.items():
            try:
                packed_paths[source] = pack_authors_to_disk(source, settings, data_dir)
            except Exception as e:
                print(f"Exception {e}")

    # (path, is_twitter, batch_size)
    tasks_meta = [
        (str(packed_paths[s]), s=='twitter', CONFIG[s].get('batch_size', 256))
        for s in CONFIG
    ]

    total_rows = 0
    total_batches = 0
    for path, _, bs in tasks_meta:
        n = pq.read_metadata(path).num_rows
        total_rows += n
        total_batches += (n + bs - 1) // bs
    print(f"Counted {total_rows} rows across all datasets", flush=True)

    def producer():
        """Generator to get batches of texts from the data."""
        for path, is_tw, bs in tasks_meta:
            pf = pq.ParquetFile(path)
            for rb in pf.iter_batches(batch_size=bs, columns=["text"]):
                yield (rb.column("text").to_pylist(), is_tw)

    n_workers = 4
    max_inflight = n_workers * 3  

    total = Counter()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as ex:
        inflight = set() # Unfinished batches
        pbar = tqdm(total=total_batches, desc="Compiling Global Word Counts")
        for item in producer():
            inflight.add(ex.submit(count_batch, item))
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

    print("Cleaning up temporary packed files...", flush=True)
    for path in packed_paths.values():
        if os.path.exists(path):
            os.remove(path)