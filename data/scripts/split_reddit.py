import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq 
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    dataset = ds.dataset('../reddit_chunks.parquet', format='parquet')

    table = dataset.to_table(columns=['author'])
    authors = pc.unique(table.column('author')).to_pylist()

    train_authors, val_authors = train_test_split(
        authors,
        train_size=0.9,
        shuffle=True,
        random_state=42
    )

    train_ds = dataset.to_table(filter=pc.field('author').isin(train_authors))
    val_ds = dataset.to_table(filter=pc.field('author').isin(val_authors))

    pq.write_table(train_ds, '../reddit_train.parquet')
    pq.write_table(val_ds, '../reddit_val.parquet')


