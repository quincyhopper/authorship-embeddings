import pandas as pd

def convert_to_parquet(data_path, output_path):

    df = pd.read_csv(data_path)
    df = df[['id', 'text']]
    df = df.rename(columns={'id': 'author'})
    df['doc_id'] = [f"blog_{id}" for id in range(len(df))]
    df['source'] = ['blog'] * len(df)

    df.to_parquet(output_path)
    print(f"Successfully saved {data_path} to {output_path}")

if __name__ == '__main__':
    convert_to_parquet(
        'data/blogtext.csv',
        'data/blogtext_raw.parquet'
    )