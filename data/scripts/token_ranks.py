import argparse
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from utils import load_rank_map

PROTECTED_RANK = -1

def build_rank_lookup(tokenizer, rank_map, oov_rank):
    """Create a flat 1D array where each index is the rank of the corresponding token id"""

    vocab_size = len(tokenizer)
    arr = np.full(vocab_size, fill_value=oov_rank, dtype=np.int32) # Init everything with OOV

    special_ids = {
        tokenizer.bos_token_id,  # <s>
        tokenizer.eos_token_id,  # </s>
        tokenizer.pad_token_id,  # <pad>
        tokenizer.unk_token_id,  # <unk>
    }

    # Special tokens get rank -1
    for special_id in special_ids:
        if special_id is not None:
            arr[special_id] = PROTECTED_RANK

    for id, rank in rank_map.items():
        id = int(id)
        if id < vocab_size:
            arr[id] = rank

    return arr

def _map_batch(batch, lookup_table):
    input_ids = np.array(batch['input_ids'], dtype=np.int32)
    ranks = lookup_table[input_ids]

    return {'ranks': ranks.tolist()}

def main(input_path, output_path, counts_path):

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens({"additional_special_tokens": ["<u>", "<h>"]})
    rank_map, oov_rank = load_rank_map(counts_path)
    lookup_table = build_rank_lookup(tokenizer, rank_map, oov_rank)

    ds = load_dataset('parquet', data_files=[input_path], split='train')
    ds = ds.map(
        _map_batch,
        batched=True,
        batch_size=4096,
        fn_kwargs={'lookup_table': lookup_table},
        num_proc=4,
        desc="Counting tokens"
    )

    ds.to_parquet(output_path)
    print(f"Wrote {output_path}", flush=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Add a ranks column to a dataset of chunks.")
    parser.add_argument(
        '--input',
        required=True,
        type=str,
        help="Input path for which to derive ranks."
    )
    parser.add_argument(
        '--counts',
        required=True,
        type=str,
        help='Path to counts json file.'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path to save the new file.'
    )
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        counts_path=args.counts
    )

