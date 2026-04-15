import torch
import pandas as pd
import numpy as np
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from model import ModelWrapper

def load_model(device):
    # Load model
    ckpt = torch.load('checkpoints/star-epoch=99-val_loss=4.46.ckpt', map_location='cuda')
    state_dict = ckpt['state_dict']

    # Strip the "model." Lightning prefix
    prefix = "model."
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    model = ModelWrapper("roberta-large").to(device)
    model.load_state_dict(stripped, strict=True)
    
    return model

@torch.no_grad()
def generate_embeddings(df: pd.DataFrame, model, device):
    model.eval()

    # Stack input ids and attention masks
    ids = torch.tensor(np.stack(df['input_ids'].values))
    mask = torch.tensor(np.stack(df['attention_mask'].values))

    # Make dataset and loader for generating embeddings
    dataset = TensorDataset(ids, mask)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Generate embeddings
    all_embeddings = []
    for ids, mask in loader:
        ids, mask = ids.to(device), mask.to(device)
        emb = model(ids, mask)
        all_embeddings.append(emb.cpu().numpy())

    # Stack embeddings
    X = np.concatenate(all_embeddings, axis=0)

    return normalize(X, norm='l2')

def get_train_test_indices(y, k, n, author_counts):
    """
    Args:
        y: array of encoded authors.
        k: support level.
        n: author level.
        author_counts: number of texts per author. 

    Returns:
        train indices and test indices
    """

    # Authors that have at least k texts
    valid_authors = np.where(author_counts >= k)[0]

    # Skip current config if not enough valid authors
    if n > len(valid_authors):
        return None, None

    # 1. Randomly select N authors
    selected_authors = np.random.choice(valid_authors, n, replace=False)

    train_idx = []
    test_idx = []

    # 2. Randomly select n_support texts from the sampled authors
    for auth in selected_authors:
        auth_indices = np.where(y==auth)[0]
        selected_texts = np.random.choice(auth_indices, k, replace=False)

        # One text is the test and the others are the train
        train_idx.extend(selected_texts[:-1])
        test_idx.append(selected_texts[-1])

    return train_idx, test_idx

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet('data/reddit_val.parquet')

    model = load_model(device)

    le = LabelEncoder()
    y = le.fit_transform(df['author'])
    author_counts = np.bincount(y) # Count number of texts per author
    X = generate_embeddings(df, model, device)

    support_levels = [2, 3, 4, 9]
    author_levels = [10, 20, 50, 100, 250, 500, 1000]
    combos = list(product(support_levels, author_levels))

    results = []
    for n_support, n_authors in combos:

        trial_accuracies = []
        for trial in range(100):
            train_idx, test_idx = get_train_test_indices(y, n_support, n_authors, author_counts)

            if train_idx is None:
                break # Skip this combo if not enough authors

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            trial_accuracies.append(acc)
        
        results.append({
            'n_authors': n_authors,
            'n_support': n_support,
            'mean_acc': np.mean(trial_accuracies),
            'std_acc': np.std(trial_accuracies)
        })

    results_df = pd.DataFrame(results)
    pivot_table = results_df.pivot(index='n_authors', columns='n_support', values='mean_acc')
    print(pivot_table)