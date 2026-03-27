import glob
import os
import pyarrow.parquet as pq
import pyarrow

if __name__ == "__main__":
    data_dir = ".."
    raw_files = glob.glob(os.path.join(data_dir, "*raw.parquet"))
    
    results = {}

    for filepath in raw_files:
        filename = os.path.basename(filepath)

        try:
            pfile = pq.ParquetFile(filepath)
            rows = pfile.metadata.num_rows
            cols = pfile.metadata.num_columns
            unique_authors = set()

            for batch in pfile.iter_batches(columns=['author']):
                unique_authors.update(batch.to_pydict()['author'])

            results[filename] = {'cols': cols, 'rows': rows, 'authors': len(unique_authors)}
            print(f"Processed {filename}: {rows:,} rows, {len(unique_authors):,} authors")

        except pyarrow.lib.ArrowInvalid:
            print(f"Skipping {filename}: Invalid or corrupted Parquet file.")
        except Exception as e:
            print(f"An unexpected error occurred with {filename}: {e}")

    total_rows = sum([info['rows'] for info in results.values()])
    total_authors = sum([info['authors'] for info in results.values()])
    print(f"\nTotal rows: {total_rows:,}")
    print(f"\nTotal authors: {total_authors:,}")
