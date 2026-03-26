import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import random

if __name__ == "__main__":
    filepath = 'data/gutenberg_chunks.parquet'
    output_path = 'data/gutenberg_train.parquet'

    pf = pq.ParquetFile(filepath)

    unique_authors = set()

    for batch in pf.iter_batches(columns=['author']):
        unique_authors.update(batch.to_pydict()['author'])

    n_authors = len(unique_authors)

    random.seed(42)
    if n_authors >= 1270:
        samples = set(random.sample(list(unique_authors), 1270))
    else:
        raise ValueError("Not enough authors found")
    
    writer = None

    print(f"Filtering {n_authors} authors...")
    for batch in pf.iter_batches():
        # Convert batch to table to use filtering logic
        table = pa.Table.from_batches([batch])
        
        # Create a boolean mask: True if author is in our sample
        mask = pc.is_in(table['author'], value_set=pa.array(list(samples)))
        filtered_table = table.filter(mask)
        
        if filtered_table.num_rows > 0:
            if writer is None:
                # Initialize writer with the schema of the first filtered table
                writer = pq.ParquetWriter(output_path, filtered_table.schema)
            writer.write_table(filtered_table)

    if writer:
        writer.close()

