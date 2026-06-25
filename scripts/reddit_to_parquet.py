"""
Before running this script, make sure you have downloaded and unzipped the Reddit corpus file via:
```
wget -c "https://huggingface.co/datasets/webis/tldr-17/resolve/main/data/corpus-webis-tldr-17.zip?download=true" -O reddit.zip
unzip reddit.zip
rm twitter.zip
```

NOTE: This script must be ran on a high memory node
"""

from datasets import load_dataset

def prepare_reddit_dataset(input_json, output_path):
    print(f"Loading local JSON: {input_json}...")
    
    # Use 'json' loader for the local file. 
    ds = load_dataset("json", data_files=input_json, split="train", streaming=False)

    print("Processing and saving to Parquet")
    
    ds = ds.select_columns(['author', 'body'])
    ds = ds.rename_column('body', 'text')
    
    # Add metadata
    ds = ds.map(lambda ex, idx: {
        **ex, 
        "doc_id": f"reddit_{idx}", 
        "source": "reddit"
    }, with_indices=True)
    
    ds.to_parquet(output_path)

    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    prepare_reddit_dataset("data/corpus-webis-tldr-17.json", "data/reddit_raw.parquet")