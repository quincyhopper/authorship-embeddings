"""
This script is used for making train and validation splits from the chunked files (either *_chunks.parquet or chunks.parquet). By default, we sample 256 authors per corpus for validation and 10,000 authors from gutenberg for training.
"""

import duckdb
from pathlib import Path

def split_datasets(data_dir: Path, authors_per_val: int=256, gutenberg_train_authors: int=1270, seed: int=42):
    """Make train and validation splits by sampling a number of authors from each corpus."""
    
    # Get individual chunk files or default to master chunk file
    chunk_files = list(data_dir.glob('*_chunks.parquet'))
    if not chunk_files:
        master_file = data_dir / 'chunks.parquet'
        if master_file.exists():
            chunk_files.append(master_file)
        else:
            raise FileNotFoundError(f"Could not find chunked files in {data_dir}")
        
    input_paths_str = ", ".join([f"'{str(f)}'" for f in chunk_files])
    train_output_path = data_dir / 'train.parquet'
    val_output_path = data_dir / 'val.parquet'

    con = duckdb.connect()

    # Build validation set once and then reuse it to filter out the traing set
    con.execute(f"""
        CREATE TEMP TABLE val_authors AS
        WITH unique_authors AS (
            SELECT DISTINCT source, author
            FROM read_parquet([{input_paths_str}])
        ),
        ranked AS (
            SELECT source, author,
                   ROW_NUMBER() OVER (
                       PARTITION BY source
                       ORDER BY HASH(CAST(author AS VARCHAR) || '{seed}')
                   ) AS rn
            FROM unique_authors
        )
        SELECT source, author FROM ranked WHERE rn <= {authors_per_val};
    """)

    # VALIDATION SET
    print(f"Compiling validation dataset with {authors_per_val} authors per corpus...")
    con.execute(f"""
        COPY (
            WITH chunks_indexed AS (
                SELECT *, ROW_NUMBER() OVER () AS global_rn
                FROM read_parquet([{input_paths_str}])
            ),
            ranked AS (
                SELECT c.* EXCLUDE(global_rn),
                       ROW_NUMBER() OVER (
                           PARTITION BY c.source, c.author
                           ORDER BY HASH(CAST(c.global_rn AS VARCHAR) || '{seed}')
                       ) AS chunk_rn
                FROM chunks_indexed c
                JOIN val_authors v ON c.source = v.source AND c.author = v.author
            )
            SELECT * EXCLUDE(chunk_rn) FROM ranked
        ) TO '{val_output_path}' (FORMAT parquet, COMPRESSION snappy);
    """)
    
    # TRAINING SET
    print("Compiling training partition from remaining authors...")
    con.execute(f"""
        COPY (
            WITH train_candidates AS (
                SELECT u.source, u.author,
                       ROW_NUMBER() OVER (
                           PARTITION BY u.source
                           ORDER BY HASH(CAST(u.author AS VARCHAR) || '{seed}')
                       ) AS rn
                FROM (SELECT DISTINCT source, author FROM read_parquet([{input_paths_str}])) u
                LEFT JOIN val_authors v ON u.source = v.source AND u.author = v.author
                WHERE v.author IS NULL
            ),
            train_authors AS (
                SELECT source, author FROM train_candidates
                WHERE source != 'gutenberg' OR rn <= {gutenberg_train_authors}
            )
            SELECT c.*
            FROM read_parquet([{input_paths_str}]) c
            JOIN train_authors t ON c.source = t.source AND c.author = t.author
        ) TO '{train_output_path}' (FORMAT parquet, COMPRESSION snappy);
    """)
    
    print("\n" + "="*55)
    print(f"{"PARTITION SPLIT REPORT":^55}")
    print("="*55)
    
    val_counts = con.execute(f"""
        SELECT source, COUNT(*), COUNT(DISTINCT author)
        FROM read_parquet('{str(val_output_path)}')
        GROUP BY source
        ORDER BY source;
    """).fetchall()
    
    train_counts = con.execute(f"""
        SELECT source, COUNT(*), COUNT(DISTINCT author)
        FROM read_parquet('{str(train_output_path)}')
        GROUP BY source
        ORDER BY source;
    """).fetchall()
    
    con.close()

    print("\nValidation Partition (val.parquet):")
    print(f"{"Corpus":<12} | {"Total Chunks":<12} | {"Unique Authors":<14}")
    print("-" * 48)
    for src, chunks, auths in val_counts:
        print(f"{src:<12} | {chunks:<12,} | {auths:<14,}")
        
    print("\nTraining Partition (train.parquet):")
    print(f"{"Corpus":<12} | {"Total Chunks":<12} | {"Unique Authors":<14}")
    print("-" * 48)
    for src, chunks, auths in train_counts:
        print(f"{src:<12} | {chunks:<12,} | {auths:<14,}")
        
    print("="*55)

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    split_datasets(
        data_dir, 
        authors_per_val=256,
        gutenberg_train_authors=1270,
        seed=42
        )