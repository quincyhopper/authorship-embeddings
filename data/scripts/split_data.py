"""
This script is used for making train and validation splits from the chunked files (either *_chunks.parquet or chunks.parquet). By default, we sample 256 authors per corpus for validation and 10,000 authors from gutenberg for training.
"""

import duckdb
from pathlib import Path

def split_datasets(data_dir: Path, authors_per_corpus: int=256, gutenberg_train_authors: int=10_000, seed: int=42):
    """Make train and validation splits by sampling a number of authors from each corpus."""
    
    # Get individual chunk files or default to master chunk file
    chunk_files = list(data_dir.glob('*_chunks.parquet'))
    if not chunk_files:
        master_file = data_dir / 'chunks.parquet'
        if master_file.exists():
            chunk_files[master_file]
        else:
            raise FileNotFoundError(f"Could not find chunked files in {data_dir}")
        
    input_paths_str = ", ".join([f"'{str(f)}'" for f in chunk_files])
    train_output_path = data_dir / 'train.parquet'
    val_output_path = data_dir / 'val.parquet'

    con = duckdb.connect()

    # VALIDATION SET
    con.execute(f"""
        COPY (
            WITH unique_authors AS (
                SELECT DISTINCT source, author
                FROM read_parquet([{input_paths_str}])
            ),
            ranked_authors AS (
                SELECT source, 
                       author,
                       ROW_NUMBER() OVER (
                           PARTITION BY source 
                           ORDER BY HASH(CAST(author AS VARCHAR) || '{seed}')
                       ) as author_rn
                FROM unique_authors
            ),
            val_authors AS (
                SELECT source, author
                FROM ranked_authors
                WHERE author_rn <= {authors_per_corpus}
            ),
            chunks_indexed AS (
                -- Generates a stable global index to allow reproducible chunk shuffling
                SELECT *, ROW_NUMBER() OVER() as global_rn
                FROM read_parquet([{input_paths_str}])
            ),
            val_chunks_ranked AS (
                SELECT c.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY c.source, c.author 
                           ORDER BY HASH(c.global_rn || '{seed}')
                       ) as chunk_rn
                FROM chunks_indexed c
                JOIN val_authors v 
                  ON c.source = v.source 
                 AND c.author = v.author
            )
            SELECT 
                * EXCLUDE(global_rn, chunk_rn)
            FROM val_chunks_ranked
            WHERE chunk_rn <= 16
        ) TO '{str(val_output_path)}' (FORMAT parquet, COMPRESSION snappy);
    """)
    
    # TRAINING SET
    print("Compiling training partition (routing remaining authors)...")
    con.execute(f"""
        COPY (
            WITH unique_authors AS (
                SELECT DISTINCT source, author
                FROM read_parquet([{input_paths_str}])
            ),
            ranked_authors AS (
                SELECT source, 
                       author,
                       ROW_NUMBER() OVER (
                           PARTITION BY source 
                           ORDER BY HASH(CAST(author AS VARCHAR) || '{seed}')
                       ) as author_rn
                FROM unique_authors
            ),
            val_authors AS (
                SELECT source, author
                FROM ranked_authors
                WHERE author_rn <= {authors_per_corpus}
            ),
            train_candidates AS (
                SELECT r.source, r.author, 
                       ROW_NUMBER() OVER (
                           PARTITION BY r.source 
                           ORDER BY r.author_rn
                       ) as train_rn
                FROM ranked_authors r
                LEFT JOIN val_authors v 
                  ON r.source = v.source AND r.author = v.author
                WHERE v.author IS NULL
            ),
            train_authors AS (
                SELECT source, author
                FROM train_candidates
                WHERE source != 'gutenberg' OR train_rn <= {gutenberg_train_authors}
            )
            SELECT c.*
            FROM read_parquet([{input_paths_str}]) c
            JOIN train_authors t 
              ON c.source = t.source 
             AND c.author = t.author
        ) TO '{str(train_output_path)}' (FORMAT parquet, COMPRESSION snappy);
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
    data_dir = Path(__file__).resolve().parent.parent
    split_datasets(
        data_dir, 
        authors_per_corpus=256,
        gutenberg_train_authors=10_000,
        seed=42
        )