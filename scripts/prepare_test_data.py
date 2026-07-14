"""
This script is for preparing the test data from the Lambda G paper. 

The quanteda corpus objects were flattened to data frames in R and saved. With these, we can filter duplicate authors
"""

import pyreadr
import pandas as pd

if __name__ == "__main__":

    train_data = pyreadr.read_r('data/lambdag_data/flat_train_data.rds')[None]
    test_data = pyreadr.read_r('data/lambdag_data/flat_test_data.rds')[None]
    train_problems = pyreadr.read_r('data/lambdag_data/training_problems.rds')[None]
    test_problems = pyreadr.read_r('data/lambdag_data/test_problems.rds')[None]

    valid_corpora = [
        'All-the-news', 'IMDB', 'TripAdvisor', 'Wiki', 'ACL', 'Amazon',
        'The Apricity', 'Enron', 'Perverted Justice', 'StackExchange',
        'The Telegraph', 'Yelp'
    ]

    train_filtered = train_data[train_data['corpus'].isin(valid_corpora)].copy()
    test_filtered = test_data[test_data['corpus'].isin(valid_corpora)].copy()

    # Combine train and test to find any duplicate authors
    combined = pd.concat([train_filtered, test_filtered], ignore_index=True)
    duplicates = combined.duplicated(subset=['text'], keep=False)

    # Calculate how many bad authors per corpus
    bad_authors_by_corpus = (
        combined[duplicates]
        .dropna(subset=['author'])
        .groupby('corpus')['author']
        .nunique()
        .reindex(valid_corpora, fill_value=0)
    )

    # Find authors who have duplicate texts
    bad_authors = combined.loc[duplicates, 'author'].dropna().unique()
    print(f"-> Found {len(bad_authors)} unique authors involved in duplications.")

    # Remove all texts by these duplicate authors
    if len(bad_authors) > 0:
        print("Removing bad authors from datasets...")

        final_train = train_filtered[~train_filtered['author'].isin(bad_authors)]
        final_test = test_filtered[~test_filtered['author'].isin(bad_authors)]

        final_train_problems = train_problems[
            (~train_problems['known_author'].isin(bad_authors)) & 
            (~train_problems['unknown_author'].isin(bad_authors))
        ]

        final_test_problems = test_problems[
            (~test_problems['known_author'].isin(bad_authors)) & 
            (~test_problems['unknown_author'].isin(bad_authors))
        ]
    else:
        final_train = train_filtered
        final_test = test_filtered
        final_train_problems = train_problems
        final_test_problems = test_problems

    print("Saving cleaned data to CSV...")
    final_train.to_csv('data/lambdag_data/clean_train.csv', index=False)
    final_test.to_csv('data/lambdag_data/clean_test.csv', index=False)
    final_train_problems.to_csv('data/lambdag_data/clean_train_problems.csv', index=False)
    final_test_problems.to_csv('data/lambdag_data/clean_test_problems.csv', index=False)

    print("\nPreparation Complete!")

    print("\nUnique bad authors found per corpus:")
    for corpus, count in bad_authors_by_corpus.items():
        print(f"  - {corpus:<20} : {count} author(s)")

    print(f"Train Set: {len(train_data):,} rows -> {len(final_train):,} rows")
    print(f"Test Set:  {len(test_data):,} rows -> {len(final_test):,} rows")
    print(f"Train Problems: {len(train_problems):,} rows -> {len(final_train_problems):,} rows")
    print(f"Test Problems:  {len(test_problems):,} rows -> {len(final_test_problems):,} rows")