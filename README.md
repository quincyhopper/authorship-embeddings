# Authorship embeddings

## Replication of STAR
This section covers all the steps to replicate STAR

### How to download the data

#### Standardized Project Gutenberg
Clone the repository via:
```
git clone https://github.com/pgcorpus/gutenberg
cd gutenberg
```

And run the following commands:
```
python get_data.py
python process_data.py
```

I ran these commands on 18/03/2026. I also modified the gutenberg repo to skip the counts and tokens processes.

#### Reddit TL;DR-17
The Reddit dataset can be found at https://huggingface.co/datasets/webis/tldr-17. To download it, run:
```
wget -c "https://huggingface.co/datasets/webis/tldr-17/resolve/main/data/corpus-webis-tldr-17.zip?download=true" -O reddit.zip
unzip reddit.zip
rm reddit.zip
```

This will create a 19GB JSON file called `corpus-webis-tldr-17.json`. Convert this to a parquet file via:
```
uv run reddit_to_parquet.py
```

#### Twitter
The twitter dataset was originally scraped via the research API. This is no longer available, so we use https://archive.org/details/twitter_cikm_2010. Download this file via:
```
wget -c "https://archive.org/download/twitter_cikm_2010/twitter_cikm_2010.zip" -O twitter.zip
unzip twitter.zip
rm twitter.zip
```

Convert this to a parquet file via:
```
uv run misc/twitter_to_parquet.py
```

#### Blogs
The blog dataset can be found at https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus. Download it manually and convert it to a parquet file via:

```
uv run misc/blog_to_parquet.py
```