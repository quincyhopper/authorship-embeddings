"""
This script is for testing a model on the lambda g corpora and saving predictions.
"""
import pyreadr 
import torch
import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from utils import load_model, generate_embeddings, build_siamese_pair, load_rank_map, calculate_masking_threshold, create_rank_tensor

def load_data():
    # Load raw data
    train_data = pyreadr.read_r('data/lambdag_data/flat_train_data.rds')[None]
    test_data = pyreadr.read_r('data/lambdag_data/flat_test_data.rds')[None]
    train_probs = pyreadr.read_r('data/lambdag_data/training_problems.rds')[None]
    test_probs = pyreadr.read_r('data/lambdag_data/test_problems.rds')[None]

    # Filter Reddit and Blogs
    train_data = train_data[~train_data['corpus'].isin(['Reddit', "Koppel's Blogs"])]
    test_data = test_data[~test_data['corpus'].isin(['Reddit', "Koppel's Blogs"])]

    return train_data, test_data, train_probs, test_probs

def get_problem_texts(data: pd.DataFrame, probs: pd.DataFrame, corpus_name: str=""):
    """
    Fetch the Q and K texts from the data using the problems dataset.
    
    Args:
        data: dataframe containing the texts.
        probs: dataframe containing the problem pairs.
        corpus_name: name of the corpus (just used for printing).

    Returns:
        lists of known_texts, unknown_texts, labels, and metadata (K and Q authors).
    """
    known_texts = []
    questioned_texts = []
    labels = []
    metadata = []
    
    missing_k = 0
    missing_q = 0
    multiple_k = 0
    multiple_q = 0
    success_count = 0

    for i, row in probs.iterrows():
        k_author = row['known_author']
        q_author = row['unknown_author']

        k_matches = data.loc[(data['author'] == k_author) & (data['texttype'] == 'known'), 'text']
        q_matches = data.loc[(data['author'] == q_author) & (data['texttype'] == 'unknown'), 'text']

        # Track anomalies for visibility
        if len(k_matches) == 0: missing_k += 1
        if len(k_matches) > 1:  multiple_k += 1
        if len(q_matches) == 0: missing_q += 1
        if len(q_matches) > 1:  multiple_q += 1

        # Skip only if text is genuinely missing from this slice
        if len(k_matches) == 0 or len(q_matches) == 0:
            continue

        # Grab the first available text chunk if multiple exist
        k_text = k_matches.iloc[0]
        q_text = q_matches.iloc[0]
        
        label = k_author == q_author
        
        known_texts.append(k_text)
        questioned_texts.append(q_text)
        labels.append(label)
        metadata.append({
            'known_author': k_author,
            'unknown_author': q_author,
        })
        success_count += 1

    print(f"\n--- Data Mapping Diagnostics: {corpus_name} ---")
    print(f"  Total Problems Checked: {len(probs)}")
    print(f"  Successfully Matched:   {success_count}")
    print(f"  Missing Known Texts:    {missing_k}")
    print(f"  Missing Questioned:     {missing_q}")
    print(f"  Multiple Known Matches: {multiple_k} (Resolved using first match)")
    print(f"  Multiple Questioned:    {multiple_q} (Resolved using first match)")

    return known_texts, questioned_texts, labels, metadata

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help="Filepath to the model for testing."
    )
    parser.add_argument(
        '--counts',
        type=str,
        default=None,
        help='Path to counts file used for masking'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='Masking strategy.'
    )
    parser.add_argument(
        '--masking_value',
        type=int,
        help="Masking value"
    )
    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help="Filepath to save the csv file."
    )
    args = parser.parse_args()

    # Load data
    print("Loading data", flush=True)
    train_data, test_data, train_probs, test_probs = load_data()

    print("Loading model", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    #model = DummyModel()

    # Load tokenizer and any
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ["<u>", "<h>"]})

    if args.counts is not None:
        rank_map, oov_rank = load_rank_map(args.counts)
        rank_tensor = create_rank_tensor(rank_map, oov_rank, tokenizer)
        masking_threshold = calculate_masking_threshold(args.counts, strategy=args.strategy, value=args.masking_value)
    else:
        rank_tensor = None
        masking_threshold = None

    corpora = [
        'All-the-news',
        'IMDB',
        'TripAdvisor',
        'Wiki',
        'ACL',
        'Amazon',
        'The Apricity',
        'Enron',
        'Perverted Justice',
        'StackExchange',
        'The Telegraph',
        'Yelp'
    ]
    
    results = {}
    all_test_predictions = []  # Container to hold metadata + predictions across all corpora
    
    for corpus in corpora:
        print(f"\n================ Processing Corpus: {corpus} ================")
        
        # TRAINING SET EMBEDDINGS
        train_corp = train_data[train_data['corpus'] == corpus]
        local_train_probs = train_probs[train_probs['corpus'] == corpus]

        if len(local_train_probs) == 0:
            print(f"Skipping {corpus}: No training problems found.")
            continue

        known_train, questioned_train, labels_train, _ = get_problem_texts(train_corp, local_train_probs, corpus)
        
        if not labels_train:
            print(f"Skipping {corpus}: Problems could not be matched to available texts.")
            continue

        print(f"Generating training embeddings for {len(labels_train)} pairs...")
        k_emb_train = generate_embeddings(known_train, model, tokenizer, device, rank_tensor, masking_threshold, batch_size=64)
        q_emb_train = generate_embeddings(questioned_train, model, tokenizer, rank_tensor, masking_threshold, device, batch_size=64)
        
        X_train = build_siamese_pair(k_emb_train, q_emb_train).cpu().numpy()
        y_train = np.array(labels_train, dtype=int)

        # TEST SET EMBEDDINGS
        test_corp = test_data[test_data['corpus'] == corpus]
        local_test_probs = test_probs[test_probs['corpus'] == corpus]

        if len(local_test_probs) == 0:
            print(f"Skipping Evaluation for {corpus}: No test problems found.")
            continue

        known_test, questioned_test, labels_test, test_meta = get_problem_texts(test_corp, local_test_probs, corpus)
        
        if not labels_test:
            print(f"Skipping Evaluation for {corpus}: Test problems could not be matched to texts.")
            continue

        print(f"Generating test embeddings for {len(labels_test)} pairs...")
        k_emb_test = generate_embeddings(known_test, model, tokenizer, device, rank_tensor, masking_threshold, batch_size=64)
        q_emb_test = generate_embeddings(questioned_test, model, tokenizer, device, rank_tensor, masking_threshold, batch_size=64)
        
        X_test = build_siamese_pair(k_emb_test, q_emb_test).cpu().numpy()
        y_test = np.array(labels_test, dtype=int)

        # LOGISTIC REGRESSION FOR PREDICTIONS
        print(f"Training Logistic Regression for {corpus}...")
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        
        # Zip metadata, ground truth, and predictions together
        for meta, y_true_i, y_pred_i in zip(test_meta, y_test, y_pred):
            meta['corpus'] = corpus
            meta['ground_truth'] = int(y_true_i)
            meta['predicted'] = int(y_pred_i)
            meta['correct'] = int(y_true_i == y_pred_i)
            all_test_predictions.append(meta)

        report = classification_report(y_test, y_pred)
        results[corpus] = report
        print(f"\nResults for {corpus}:")
        print(report)

    if all_test_predictions:
        df_preds = pd.DataFrame(all_test_predictions)
        df_preds.to_csv(args.output, index=False)
        print(f"\nSaved all test predictions to {args.output}")

    print("\n================ All Evaluation Rounds Completed ================")