import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Define files to process mapping to their output names
FILE_MAPPING = {
    'data/training_set_tweets.txt': 'data/twitter_train_raw.parquet',
    'data/test_set_tweets.txt': 'data/twitter_test_raw.parquet'
}

# Have to specify columns as there aren't any in the txt files
COLS = ['author', 'tweet_id', 'text', 'created_at']

SCHEMA = pa.schema([
    ('author', pa.string()),
    ('text', pa.string()),
    ('doc_id', pa.string()),
    ('source', pa.string()),
])

def process_twitter_data():
    for input_file, output_parquet in FILE_MAPPING.items():
        if not os.path.exists(input_file):
            print(f"Skipping {input_file}: File not found.")
            continue
            
        print(f"Converting {input_file} to {output_parquet}...")
        
        reader = pd.read_csv(
            input_file, 
            sep='\t', 
            names=COLS, 
            header=None, 
            quoting=3, 
            on_bad_lines='skip', 
            chunksize=50_000,
            encoding='utf-8'
        )

        with pq.ParquetWriter(output_parquet, SCHEMA) as writer:
                for chunk in reader:
                    # 1. Rename tweet_id to doc_id and filter columns
                    processed_chunk = chunk[['author', 'tweet_id', 'text']].rename(
                        columns={'tweet_id': 'doc_id'}
                    )
                    
                    # Convert columns to strings
                    processed_chunk['doc_id'] = processed_chunk['doc_id'].astype(str)
                    processed_chunk['author'] = processed_chunk['author'].astype(str)
                    processed_chunk['text'] = processed_chunk['text'].fillna('').astype(str)
                    
                    # 3. Add static source metadata
                    processed_chunk['source'] = 'twitter'
                    
                    # 4. Write to Parquet
                    table = pa.Table.from_pandas(processed_chunk, schema=SCHEMA, preserve_index=False)
                    writer.write_table(table)

if __name__ == "__main__":
    process_twitter_data()