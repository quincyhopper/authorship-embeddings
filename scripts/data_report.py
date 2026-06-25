import argparse
import duckdb
from pathlib import Path

def produce_report(files: list[Path], title: str, total_label: str, row_label: str):
    """Executes a DuckDB aggregation over an explicit list of parquet files."""
    if not files:
        return
        
    file_strings = [str(f) for f in files]
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
    print(f"{title:^60}")
    print("="*60)
    
    for filename, row_count, unique_authors in report_data:
        if filename is not None:
            short_name = Path(filename).name
            print(f"File: {short_name}")
            print(f"  ├── {row_label}: {row_count:,}")
            print(f"  └── Unique Authors: {unique_authors:,}")
            print("-" * 60)
        else:
            print(f"{total_label:^60}")
            print("="*60)
            print(f"  ├── Total {row_label}: {row_count:,}")
            print(f"  └── Unique Authors: {unique_authors:,}")
            print("="*60)

def resolve_paths(file_inputs: list[str], data_dir: Path) -> list[Path]:
    """Helper to find files whether passed as raw paths or filenames relative to data_dir."""
    resolved = []
    for f in file_inputs:
        p_obj = Path(f)
        if p_obj.exists():
            resolved.append(p_obj)
        elif (data_dir / f).exists():
            resolved.append(data_dir / f)
        else:
            raise FileNotFoundError(f"Could not find file '{f}' locally or in '{data_dir}'")
    return resolved


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    
    p = argparse.ArgumentParser(description="Generate dataset reports for arbitrary parquet files.")
    
    p.add_argument(
        "--multi", "-m",
        nargs="+", # 1 or more arguments
        help="Pass a space-separated list of chunk files to count."
    )
    p.add_argument(
        "--single", "-s",
        type=str,
        help="Pass a single specific chunk file to count."
    )
    
    args = p.parse_args()
    
    try:
        # Raw files are ALWAYS counted first
        raw_files = list(data_dir.glob("*raw.parquet"))
        produce_report(
            files=raw_files, 
            title="RAW FILES REPORT", 
            total_label="TOTAL RAW", 
            row_label="Documents"
        )
        
        if args.multi:
            target_files = resolve_paths(args.multi, data_dir)
            produce_report(
                files=target_files,
                title="CHUNKS REPORT (CUSTOM SELECTION)",
                total_label="TOTAL CHUNKS",
                row_label="Total Chunks"
            )
            
        elif args.single:
            target_files = resolve_paths([args.single], data_dir)
            produce_report(
                files=target_files,
                title="CHUNKS REPORT (SINGLE FILE)",
                total_label="TOTAL CHUNKS",
                row_label="Total Chunks"
            )
            
        else:
            # No arguments = count chunks.parquet file
            default_single = data_dir / 'chunks.parquet'
            if default_single.exists():
                produce_report([default_single], "CHUNKS REPORT (SINGLE FILE)", "TOTAL CHUNKS", "Total Chunks")
            else:
                fallback_multi = list(data_dir.glob("*_chunks.parquet"))
                produce_report(fallback_multi, "CHUNKS REPORT (SPLIT FILES)", "TOTAL CHUNKS", "Total Chunks")
                
    except Exception as e:
        print(f"Exception: {e}")