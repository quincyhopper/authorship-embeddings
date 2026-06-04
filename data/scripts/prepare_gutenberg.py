"""
This script unifies the gutenberb txt files into one parquet file, which is consistent with the other datasets. 
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq 
from tqdm import tqdm

TEXT_DIR = 'data/gutenberg/data/text/'
METADATA_PATH = 'data/gutenberg/metadata/metadata.csv'
OUTPUT_PATH = 'data/gutenberg_raw.parquet'

def get_clean_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)

    # Get only english texts
    df = df[df['language'].apply(lambda x: x == "['en']")]

    # Filter anonymous authors
    a_title_pattern = r'^(?:A|An)\s+(?!\.)'
    general_pattern = r'anonymous|unknown|anon\b|various|staff|collaborative|\band\b|&'
    combined_pattern = f"({a_title_pattern})|({general_pattern})"

    filtered_df = df[~df['author'].str.contains(combined_pattern, case=False, na=True)]

    return filtered_df

def build_raw_gutenberg_parquet():
    metadata = get_clean_metadata(METADATA_PATH)
    
    # Define a clean schema matching RAW_FEATURES exactly
    schema = pa.schema([
        ('author', pa.string()),
        ('text', pa.string()),
        ('doc_id', pa.string()),
        ('source', pa.string()),
    ])

    print(f"Compiling raw text files into {OUTPUT_PATH}...")
    
    with pq.ParquetWriter(OUTPUT_PATH, schema, compression='snappy') as writer:
        # Process in chunks of rows to minimise memory
        chunk_size = 500
        for i in tqdm(range(0, len(metadata), chunk_size)):
            batch_df = metadata.iloc[i : i + chunk_size] # Small chunk of rows from metadata
            
            authors, texts, doc_ids, sources = [], [], [], []
            for _, row in batch_df.iterrows():
                filepath = os.path.join(TEXT_DIR, f"{row['id']}_text.txt")
                
                if not os.path.exists(filepath):
                    print(f"Could not find {filepath}")
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    book_text = f.read()
                
                # Append raw records
                authors.append(str(row['author']))
                texts.append(book_text)
                doc_ids.append(f"gutenberg_{row['id']}")
                sources.append('gutenberg')
            
            if texts:
                table = pa.Table.from_arrays([authors, texts, doc_ids, sources], schema=schema)
                # Write to the file incrementally with conservative row groups
                writer.write_table(table)

if __name__ == "__main__":
    build_raw_gutenberg_parquet()
