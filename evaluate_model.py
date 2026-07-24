"""
This script is for testing a model on the lambda g corpora and saving predictions.
"""
import torch
import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from src.utils import load_model, generate_embeddings, build_siamese_pair, load_rank_map, calculate_masking_threshold, create_rank_tensor

def load_data():
    # Load raw data
    train_data = pd.read_csv('data/lambdag_data/train_corpus.csv')
    test_data = pd.read_csv('data/lambdag_data/test_corpus.csv')
    train_probs = pd.read_csv('data/lambdag_data/train_problems.csv')
    test_probs = pd.read_csv('data/lambdag_data/test_problems.csv')

    return train_data, test_data, train_probs, test_probs

def get_author_centroid(X: torch.Tensor, sample_map: list, author_list: list) -> tuple[torch.Tensor, np.ndarray]:
    """
    Calculates the average embedding (centroid) for all chunks belonging to each author.
    
    Args:
        X: tensor of shape (num_chunks, hidden_dim).
        sample_map (list): list mapping chunk index -> document index.
        author_list (list): list mapping document index -> author name.
        
    Returns:
        centroids: tensor of shape (num_unique_authors, hidden_dim).
        unique_authors: 1D array of author names corresponding to the rows in `centroids`.
    """
    # Map every chunk to its corresponding author string
    chunk_authors = np.array([author_list[doc_idx] for doc_idx in sample_map])

    # Convert author strings to contiguous integer indices
    unique_authors, author_indices = np.unique(chunk_authors, return_inverse=True)
    indices_tensor = torch.tensor(author_indices, device=X.device)

    num_authors = len(unique_authors)
    hidden_dim = X.shape[-1]

    # Vectorized summation using scatter_add_
    sums = torch.zeros((num_authors, hidden_dim), device=X.device)
    expanded_indices = indices_tensor.unsqueeze(1).expand(-1, hidden_dim)
    sums.scatter_add_(0, expanded_indices, X)

    # Divide by the count of chunks per author to calculate the mean
    counts = torch.bincount(indices_tensor).unsqueeze(1).float()
    centroids = sums / counts

    # Find the row indices for the sequence of problems
    indices = np.searchsorted(unique_authors, author_list)
    aligned = centroids[indices]

    return aligned

def get_unique_author_centroids(
    unique_authors: list,
    text_dict: dict,
    model,
    tokenizer,
    device,
    rank_tensor,
    masking_threshold,
    full_author_sequence: np.ndarray,
    desc: str = ""
) -> torch.Tensor:
    """
    Embed authors exactly once and repeat their embeddings where necessary.
    """
    # Look up non-duplicated raw text content
    unique_texts = [text_dict[auth] for auth in unique_authors]
    
    print(f"Generating unique {desc} embeddings ({len(unique_authors)} authors)...")
    emb_norm, chunk_map = generate_embeddings(
        unique_texts, model, tokenizer, device, 
        rank_tensor, masking_threshold, batch_size=128
    )
    
    # Compute centroids
    centroids = get_author_centroid(emb_norm, chunk_map, unique_authors)
    
    # Repeat centroids for each problem where necessary (avoids recomputing embeddings over and over again)
    alignment_indices = np.searchsorted(unique_authors, full_author_sequence)
    
    return centroids[alignment_indices]


def prepare_eval_split(
    data: pd.DataFrame,
    probs: pd.DataFrame,
    model,
    tokenizer,
    device,
    rank_tensor,
    masking_threshold,
    corpus_name: str,
    split_name: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Builds the siamese embeddings for all of the provided problems. 
    """
    # Author -> concatenated texts dictionary
    k_dict = data[data['texttype'] == 'known'].groupby('author')['text'].apply(lambda x: " ".join(x.astype(str))).to_dict()
    q_dict = data[data['texttype'] == 'unknown'].groupby('author')['text'].apply(lambda x: " ".join(x.astype(str))).to_dict()

    # Drop verification pairs that do not map to matching text inputs
    valid_probs = probs[
        probs['known_author'].isin(k_dict.keys()) & 
        probs['unknown_author'].isin(q_dict.keys())
    ].copy()

    if len(valid_probs) == 0:
        return None, None, None

    print(f"--- Data Mapping Diagnostics ({split_name}): {corpus_name} ---")
    print(f"  Total Problems Checked: {len(probs)}")
    print(f"  Successfully Matched:   {len(valid_probs)}")
    print(f"  Missing Pairs Skipped:  {len(probs) - len(valid_probs)}")

    # Extract unique known profiles and generate embeddings
    unique_k_authors = sorted(list(valid_probs['known_author'].unique()))
    k_centroids = get_unique_author_centroids(
        unique_k_authors, k_dict, model, tokenizer, device,
        rank_tensor, masking_threshold, valid_probs['known_author'].values, desc=f"known {split_name}"
    )

    unique_q_authors = sorted(list(valid_probs['unknown_author'].unique()))
    q_centroids = get_unique_author_centroids(
        unique_q_authors, q_dict, model, tokenizer, device,
        rank_tensor, masking_threshold, valid_probs['unknown_author'].values, desc=f"questioned {split_name}"
    )

    # Build siamese pairs
    X = build_siamese_pair(k_centroids, q_centroids).cpu().numpy()
    y = (valid_probs['known_author'].values == valid_probs['unknown_author'].values).astype(int)

    return X, y, valid_probs

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
        type=float,
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
    # model = DummyModel()

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
        
        # --- TRAINING SET GENERATION ---
        train_corp = train_data[train_data['corpus'] == corpus]
        local_train_probs = train_probs[train_probs['corpus'] == corpus]

        if len(local_train_probs) == 0:
            print(f"Skipping {corpus}: No training problems found.")
            continue

        X_train, y_train, _ = prepare_eval_split(
            train_corp, local_train_probs, model, tokenizer, device,
            rank_tensor, masking_threshold, corpus, "Train"
        )
        
        if X_train is None:
            print(f"Skipping {corpus}: Train problems could not be matched to available texts.")
            continue

        # --- TEST SET GENERATION ---
        test_corp = test_data[test_data['corpus'] == corpus]
        local_test_probs = test_probs[test_probs['corpus'] == corpus]

        if len(local_test_probs) == 0:
            print(f"Skipping Evaluation for {corpus}: No test problems found.")
            continue

        X_test, y_test, valid_test_probs = prepare_eval_split(
            test_corp, local_test_probs, model, tokenizer, device,
            rank_tensor, masking_threshold, corpus, "Test"
        )

        if X_test is None:
            print(f"Skipping Evaluation for {corpus}: Test problems could not be matched to texts.")
            continue

        # --- LOGISTIC REGRESSION CLASSIFICATION ---
        print(f"Training Logistic Regression for {corpus}...")
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        
        # Gather evaluation tracking arrays using metadata row values
        for (_, row), y_true_i, y_pred_i in zip(valid_test_probs.iterrows(), y_test, y_pred):
            all_test_predictions.append({
                'known_author': row['known_author'],
                'unknown_author': row['unknown_author'],
                'corpus': corpus,
                'ground_truth': int(y_true_i),
                'predicted': int(y_pred_i),
                'correct': int(y_true_i == y_pred_i)
            })

        report = classification_report(y_test, y_pred)
        results[corpus] = report
        print(f"\nResults for {corpus}:")
        print(report)

    if all_test_predictions:
        df_preds = pd.DataFrame(all_test_predictions)
        df_preds.to_csv(args.output, index=False)
        print(f"\nSaved all test predictions to {args.output}")

    print("\n================ All Evaluation Rounds Completed ================")