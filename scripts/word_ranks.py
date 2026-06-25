"""
Derive per-token word ranks from chunked data. Word ranks look like [1245314, 312312, 4323, ...]. This is used for masking during training. This script copies the provided parquet file (e.g. train.parquet) and just adds the new `ranks` column. 

A chunk (row) stores input_ids and word_ids boundaries. Word strings are decoded from input_ids, grouped by word_ids, and word ranks are derived from whatever word_counts.json file is present. So it is possible to change how counts are produced and then regenerate ranks WITHOUT having to re-run `chunk_datasets.py`.
"""

import argparse
import sys
from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from src.utils import load_rank_map

PROTECTED_RANK = -1

# Cache id->token list per tokenizer so we don't rebuild it on every map call.
_ID_TO_TOKEN_CACHE = {}

def get_id_to_token(tokenizer):
    """Returns a list where the index is a token id and the value is its token string. Since this is called for every batch, we avoid reconstructing the list by caching it.
    
    For example, table[0] -> '<s>'; table[42] -> 'Ġthis'.
    """
    key = id(tokenizer) # Key the tokeniser to let it survive across calls within a process
    table = _ID_TO_TOKEN_CACHE.get(key)
    if table is None:
        # First time this function is called, so construct the list and cache it
        table = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))) # len(tokenizer) is vocab size + added tokens
        _ID_TO_TOKEN_CACHE[key] = table
    return table

def derive_ranks(input_ids, word_ids, tokenizer, rank_map, oov_rank, id_to_token=None):
    """Build per-token ranks for one chunk, from input_ids + chunk-local word_ids."""
    if id_to_token is None:
        id_to_token = get_id_to_token(tokenizer)

    n = len(input_ids)
    ranks = [PROTECTED_RANK] * n

    i = 0
    while i < n:
        wid = word_ids[i]
        if wid == -1: # Special tokens (<s>, </s>, <u>, <h>) are skipped
            i += 1
            continue

        j = i + 1
        while j < n and word_ids[j] == wid:
            j += 1

        tokens = [id_to_token[t] for t in input_ids[i:j]]
        word = tokenizer.convert_tokens_to_string(tokens).strip().lower() # Reconstruct the original word

        if word and any(c.isalpha() for c in word):
            r = rank_map.get(word, oov_rank) # Look up the rank
        else:
            r = PROTECTED_RANK # Pure punctuation / numbers

        for k in range(i, j): # Write that rank for every subtoken of the word
            ranks[k] = r
        i = j # Move to next word

    return ranks

def _map_batch(batch, tokenizer, rank_map, oov_rank):
    """Run derive_ranks() on all the texts in a batch. This is applied by the .map method in materialise()."""
    id_to_token = get_id_to_token(tokenizer)
    return {
        "ranks": [
            derive_ranks(ids, wids, tokenizer, rank_map, oov_rank, id_to_token)
            for ids, wids in zip(batch["input_ids"], batch["word_ids"])
        ]
    }

def materialise(chunks_path, counts_path, out_path, keep_top_k=None, num_proc=4, batch_size=1000):
    """Main function to write the word_ranks column"""

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens({"additional_special_tokens": ["<u>", "<h>"]})

    rank_map, oov_rank = load_rank_map(counts_path, keep_top_k=keep_top_k)
    print(f"Rank map: {len(rank_map)} words (OOV rank = {oov_rank})", flush=True)

    ds = load_dataset("parquet", data_files=[chunks_path], split="train")

    ds = ds.map(
        _map_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "rank_map": rank_map, "oov_rank": oov_rank},
        desc="Deriving word ranks",
    )

    ds.to_parquet(out_path)
    print(f"Wrote {out_path}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True)
    p.add_argument("--counts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep-top-k", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args()

    materialise(args.chunks, args.counts, args.out,
                keep_top_k=args.keep_top_k, num_proc=args.num_proc,
                batch_size=args.batch_size)