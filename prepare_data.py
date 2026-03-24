import itertools
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset

def stream_tokenized_to_parquet(sorted_ds, output_file, tokenizer, batch_size=500):
    current_index = 0
    writer = None
    total_rows = len(sorted_ds)
    
    # Init generator 
    iterator = itertools.groupby(sorted_ds, key=lambda x: x['author'])
    
    finished = False
    while not finished:
        batch_authors = []
        batch_texts = []
        
        # Collect a batch of authors and their joined strings
        for _ in range(batch_size):
            try:
                author, group = next(iterator)
                n_texts = sum(1 for _ in group)
                end = current_index + n_texts
                
                texts_list = sorted_ds[current_index:end]['text']
                joined_text = " </s> </s> ".join(texts_list)
                
                batch_authors.append(author)
                batch_texts.append(joined_text)
                
                current_index = end
            except StopIteration:
                finished = True
                break

        if not batch_texts:
            break

        # Tokenise
        tokenized_batch = tokenizer(batch_texts, truncation=True, max_length=512, return_overflowing_tokens=True, padding=False)
        sample_map = tokenized_batch['overflow_to_sample_mapping']

        # Get indices of chunks that are 512 tokens
        # Store as list of (chunk_idx, author_idx)
        valid_chunks = [
            (i, sample_map[i])
            for i, ids in enumerate(tokenized_batch['input_ids']) 
            if len(ids) == 512
            ]
        
        # Count how many valid chunks each author has
        author_counts = Counter(author_idx for chunk_idx, author_idx in valid_chunks)

        # Only keep a chunk if its author has at least 16 chunks
        final_authors = []
        final_input_ids = []
        final_attention_masks = []
        for chunk_idx, author_idx in valid_chunks:
            if author_counts[author_idx] >= 16:
                final_authors.append(batch_authors[author_idx])
                final_input_ids.append(tokenized_batch['input_ids'][chunk_idx])
                final_attention_masks.append(tokenized_batch['attention_mask'][chunk_idx])

        if not final_authors:
            print(f"Skipping batch - no authors with >= 16 chunks")
            continue

        # 3. Convert to Arrow Table
        table_data = {
            'author': [str(author) for author in final_authors],
            'source': ['blog'] * len(final_authors),
            'input_ids': final_input_ids,
            'attention_mask': final_attention_masks
        }
        
        table = pa.Table.from_pydict(table_data)

        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
        
        # 4. Write to Parquet and clear RAM
        writer.write_table(table)
        
        # Explicit Cleanup
        del batch_authors, batch_texts, tokenized_batch, table
        gc.collect() 
        
        print(f"Processed through index: {current_index:,} ({(current_index/total_rows * 100):.2f}%)", flush=True)

    writer.close()
    print("\nSuccessfully saved to Parquet.")

if __name__ == "__main__":

    ds = load_dataset(path='parquet', data_files=['data/reddit_raw.parquet'], split='train')

    print("Sorting dataset...")
    ds = ds.sort('author')

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    stream_tokenized_to_parquet(ds, "data/reddit_chunks.parquet", tokenizer, batch_size=500) 