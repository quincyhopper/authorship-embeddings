import duckdb
from pathlib import Path

def count_raw(data_dir: Path):
    raw_files = list(data_dir.glob("*raw.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No files matching '*raw.parquet' found in {data_dir}")
        
    file_strings = [str(f) for f in raw_files]
    con = duckdb.connect()
    
    query = """
        SELECT 
            filename,
            COUNT(*) AS row_count,
            COUNT(DISTINCT author) AS unique_authors
        FROM read_parquet(?, filename=True)
        GROUP BY ROLLUP(filename)
        ORDER BY filename NULLS LAST;
    """
    
    report_data = con.execute(query, [file_strings]).fetchall()
    con.close()
    
    print("\n" + "="*60)
    print(f"{'RAW FILES REPORT':^60}")
    print("="*60)
    
    for filename, row_count, unique_authors in report_data:
        if filename is not None:
            short_name = Path(filename).name
            print(f"File: {short_name}")
            print(f"  ├── Documents: {row_count:,}")
            print(f"  └── Unique Authors: {unique_authors:,}")
            print("-" * 60)
        else:
            print(f"{'TOTAL RAW':^60}")
            print("="*60)
            print(f"  ├── Total Documents: {row_count:,}")
            print(f"  └── Unique Authors: {unique_authors:,}")
            print("="*60)

def count_chunks(data_dir: Path):
    chunks_path = data_dir / 'chunks.parquet'
    if not chunks_path.exists():
        raise FileNotFoundError(f"The required chunks file does not exist at: {chunks_path}")
    
    con = duckdb.connect()

    # Added ROLLUP and GROUPING to seamlessly calculate grand totals
    query = f"""
        SELECT 
            source, 
            COUNT(*) AS chunk_count, 
            COUNT(DISTINCT author) AS unique_authors,
            GROUPING(source) AS is_total
        FROM read_parquet('{chunks_path}') 
        GROUP BY ROLLUP(source)
        ORDER BY is_total ASC, chunk_count DESC;
    """

    report_data = con.execute(query).fetchall()
    con.close()

    print("\n" + "="*60)
    print(f"{'CHUNKS REPORT':^60}")
    print("="*60)
    
    for source, chunk_count, unique_authors, is_total in report_data:
        # is_total == 0 means it's a regular corpus row
        if is_total == 0:
            source_name = source if source is not None else "[Unknown Source]"
            print(f"Corpus: {source_name}")
            print(f"  ├── Total Chunks: {chunk_count:,}")
            print(f"  └── Unique Authors: {unique_authors:,}")
            print("-" * 60)
        # is_total == 1 means it's the ROLLUP grand total row
        else:
            print(f"{'TOTAL CHUNKS':^60}")
            print("="*60)
            print(f"  ├── Total Chunks: {chunk_count:,}")
            print(f"  └── Unique Authors: {unique_authors:,}")
            print("="*60)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent
    
    try:
        count_raw(data_dir)
        count_chunks(data_dir)
    except Exception as e:
        print(f"Exception {e}")
