import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

TEXT_DIR = 'data/gutenberg/data/text/'
METADATA_PATH = 'data/gutenberg/metadata/metadata.csv'
OUTPUT_PATH = 'data/gutenberg_train.parquet'

def get_author_texts(metadata: pd.DataFrame) -> pd.DataFrame:

    # Get only english texts
    df = metadata[metadata['language'].apply(lambda x: x == "['en']")]

    # Filter anonymous authors
    a_title_pattern = r'^(?:A|An)\s+(?!\.)'
    general_pattern = r'anonymous|unknown|anon\b|various|staff|collaborative|\band\b|&'
    combined_pattern = f"({a_title_pattern})|({general_pattern})"

    # Group by author and aggregate
    filtered_df = df[~df['author'].str.contains(combined_pattern, case=False, na=True)]

    return filtered_df.groupby('author')

def tokenise_and_chunk(metadata: pd.DataFrame, tokenizer):

    all_input_ids = []
    sep_tokens = tokenizer.encode(" </s> </s> ", add_special_tokens=False)

    for _, row in metadata.iterrows():

        # Get filepath
        filepath = os.path.join(TEXT_DIR, f"{row['id']}_text.txt")

        if not os.path.exists(filepath):
            print(f"Could not find {filepath}. Skipping file.")
            continue
        
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        input_ids = tokenizer.encode(text, add_special_tokens=False)
        all_input_ids.extend(input_ids)
        all_input_ids.extend(sep_tokens)

        del text

    chunks = [all_input_ids[i : i + 512] for i in range(0, len(all_input_ids), 512)]

    return chunks[1:-1]

def gen(author_texts: pd.DataFrame, tokenizer):
    for author, texts in tqdm(author_texts, desc="Processing authors..."):
        chunks = tokenise_and_chunk(texts, tokenizer)
        chunks = [chunk for chunk in chunks if len(chunk) == 512]
        if chunks and len(chunks) >= 16:
            for chunk in chunks:
                yield {
                    'author': author,
                    'source': 'gutenberg',
                    'input_ids': chunk,
                    'attention_mask': [1] * 512
                    }
        else:
            continue

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    metadata = pd.read_csv(METADATA_PATH)

    author_texts = get_author_texts(metadata)

    ds = Dataset.from_generator(gen, gen_kwargs={'author_texts': author_texts, 'tokenizer': tokenizer})
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    ds.to_parquet(OUTPUT_PATH)