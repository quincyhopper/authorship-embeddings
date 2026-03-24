import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import gc
import duckdb
import tracemalloc
from collections import Counter, defaultdict
from transformers import AutoTokenizer

INPUT_FILE       = "data/reddit_raw.parquet"
SORTED_FILE      = "data/reddit_sorted.parquet"   # intermediate; delete after run
OUTPUT_FILE      = "data/reddit_chunks.parquet"
MODEL_NAME       = "roberta-large"
CHUNK_LEN        = 512
MIN_CHUNKS       = 16
AUTHOR_BATCH     = 1000
ROW_GROUP_SIZE   = 50_000

def log_memory(label=""):
    current, peak = tracemalloc.get_traced_memory()

    current_mb = current / 10**6
    peak_mb = peak / 10**6

    print(f"--- MEMORY USAGE [{label}] ---")
    print(f"Current: {current_mb:.2f} MB")
    print(f"Peak:    {peak_mb:.2f} MB")
    print("-" * 30)

def sort_parquet(src: str, dst: str, row_group_size: int):
    duckdb.sql(f"""
        COPY (SELECT * FROM read_parquet('{src}') ORDER BY author)
        TO '{dst}' (FORMAT parquet, ROW_GROUP_SIZE {row_group_size}, COMPRESSION snappy)
    """)

def stream_tokenized_to_parquet(
        sorted_file: str,
        output_file: str,
        tokenizer,
        author_batch: int = AUTHOR_BATCH,
        chunk_len: int = CHUNK_LEN,
        min_chunks: int = MIN_CHUNKS,
):
    
    # Load the sorted dataset
    pf = pq.ParquetFile(sorted_file)

    # Init writer to None; will be assigned in the first tokenise call and then reused
    writer = None
    
    # Prepare dict of authors and their texts
    author_texts: dict[str, list[str]] = defaultdict(list)

    # Prepare list of authors
    current_author_order: list[str] = []

    # Extract information from sorted list
    total_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    # Init counter for batches
    rows_done = 0

    # Loop over ROW_GROUP_SIZE rows
    for rg_idx in range(total_row_groups):

        # Read batch
        rg = pf.read_row_group(rg_idx, columns=['author', 'text'])

        # Get authors and texts
        authors_col = rg.column('author').to_pylist()
        texts_col = rg.column('text').to_pylist()
        del rg; gc.collect()

        # Fill dict of authors and their texts
        for author, text in zip(authors_col, texts_col):
            author = str(author)

            # Skip [deleted] author in Reddit
            if author == '[deleted]' or author == "None":
                continue

            if author not in author_texts:
                current_author_order.append(author)
            author_texts[author].append(text)

            # If single author gets too many posts, process them immediately
            if len(author_texts[author]) >= 2000:
                print(f"--- Flushing partial data for prolific author: {author} ---")
                writer = tokenise_and_chunk([author], writer, author_texts, tokenizer)
                author_texts[author] = [] # Clear memory

        del authors_col, texts_col

        # Define last author
        last_author = current_author_order[-1] if current_author_order else None

        # Authors to procces; ignore last author since some of their texts might be in the next batch
        authors_done = [a for a in current_author_order if a != last_author]

        log_memory(f"stream_tokenized_to_parquet: Organising rows {rg_idx}")

        # Print information about current batch
        rows_done += pf.metadata.row_group(rg_idx).num_rows
        print(f"\nRow-group {rg_idx+1}/{total_row_groups}  "
              f"({rows_done:,}/{total_rows:,} rows)  "
              f"processing {len(authors_done)} authors…\n", flush=True)
        
        # Loop over all authors in this row group
        for start in range(0, len(authors_done), AUTHOR_BATCH):

            # Slice AUTHOR_BATCH authors
            batch = authors_done[start : start + AUTHOR_BATCH]

            # Tokenise 
            writer = tokenise_and_chunk(batch, writer, author_texts, tokenizer)

            # Cleanup
            for a in batch: 
                del author_texts[str(a)]
            gc.collect()

        #  Add last author to next batch if exists
        current_author_order = [last_author] if last_author else []
        gc.collect()

    # Tokenise all of last authors texts regardless of whether they're in this row group
    if current_author_order:
        writer = tokenise_and_chunk(current_author_order, writer, author_texts, tokenizer)

    if writer:
        writer.close()
    print("\nAll done - saved to", output_file)

def tokenise_and_chunk(batch: list[str], writer, author_texts: dict[str, list[str]], tokenizer):
    if not batch:
        return writer
    
    log_memory("tokenise_and_chunk: Pre-text joining")
    
    # Join the texts of the authors in this batch
    batch_texts = [" </s> </s> ".join(author_texts[a]) for a in batch]

    log_memory("tokenise_and_chunk: Post-text joining/pre-tokenisation")

    # Tokenise batch
    tokens = tokenizer(
        batch_texts, 
        truncation=True, 
        max_length=512, 
        return_overflowing_tokens=True, 
        padding=False
        )
    sample_map = tokens['overflow_to_sample_mapping']

    log_memory("tokenise_and_chunk: Post-tokenisation")

    # Get indices of chunks that are exactly 512 tokens
    # Store as list of (chunk_idx, author_idx)
    valid = [
        (i, sample_map[i])
        for i, ids in enumerate(tokens['input_ids'])
        if len(ids) == 512
    ]

    # Count how many times each author appears
    author_counts = Counter(author_idx for _, author_idx in valid)

    # Init lists to store final outputs
    final_authors, final_ids, final_masks = [], [], []
    for chunk_idx, author_idx in valid:
            if author_counts[author_idx] >= 16:
                final_authors.append(batch[author_idx])
                final_ids.append(tokens['input_ids'][chunk_idx])
                final_masks.append(tokens['attention_mask'][chunk_idx])

    # Clear tokeniser output
    del tokens, valid, author_counts, batch_texts; gc.collect()

    log_memory("tokenise_and_chunk: Post-final lists")

    # If no authors passed filtering, do nothing
    if not final_authors:
        return writer
    
    # Define table to write to the output file
    table = pa.table({
        "author":         [str(a) for a in final_authors],
        "source":         ["blog"] * len(final_authors),
        "input_ids":      final_ids,
        "attention_mask": final_masks,
    })

    # Define writer if this is the first batch
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression="snappy")

    # Write to parquet file
    writer.write_table(table)
    del table, final_authors, final_ids, final_masks; gc.collect()

    log_memory('tokenise_and_chunk: Post-writing')

    return writer

if __name__ == "__main__":
    tracemalloc.start()

    log_memory("Pre-sorting")

    # Sort dataset
    print('Sorting dataset', flush=True)
    sort_parquet(src=INPUT_FILE, dst=SORTED_FILE, row_group_size=ROW_GROUP_SIZE)

    log_memory("Post-sorting")

    # Load tokenizer
    print('Loading tokenizer', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log_memory('Post loading tokenizer')

    print("Processing...")
    stream_tokenized_to_parquet(
        SORTED_FILE,
        OUTPUT_FILE,
        tokenizer,
    )