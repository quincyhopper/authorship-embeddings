"""
This is a script for counting the number of documents and unique authors in the Gutenberg corpus BEFORE preprocessing is applied.
This is necessary because, before converting the Gutenberg directory into gutenberg_raw.parquet, I removed anonymous authors. 
As a result, the 'raw' gutenberg corpus isn't really raw, and it makes it difficult to present logically in the paper.
"""
import pandas as pd
from pathlib import Path

if __name__ == "__main__":

    pg_dir = Path('/mnt/iusers01/fatpou01/hum01/msc-comp-ling-2025-2026/e34316nh/scratch/authorship-embeddings/data/gutenberg')
    meta_path = pg_dir / 'metadata/metadata.csv'
    text_dir = pg_dir / 'data/text/'

    meta = pd.read_csv(meta_path)
    meta = meta[meta['language'].apply(lambda x: x == "['en']")]

    authors = set()
    docs = 0
    missing_authors = set()
    missing_docs = 0

    for _, row in meta.iterrows():

        id = int(row['id'])
        author = row['author']

        filename = f'{id}_text.txt'
        to_check = text_dir / filename

        if to_check.is_file():
            authors.add(author)
            docs += 1
        else:
            missing_authors.add(author)
            missing_docs += 1

    print(f"Unique authors: {len(authors)}")
    print(f"Documents: {docs}")
    print(f"\nMissing authors: {len(missing_authors)}")
    print(f"Missing docs: {missing_docs}")