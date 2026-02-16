import pandas as pd

def filter_authors(df:pd.DataFrame, n:int):
    """Filters authors that don't have at least n texts."""
    counts = df['author'].value_counts()
    return df[df['author'].isin(counts[counts >= n].index)]