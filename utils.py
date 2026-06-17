import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from model import ModelWrapper

def load_rank_map(
        counts_path: str, 
        keep_top_k: int | None = None
        ) -> tuple[dict, int]:
    """
    Load the word_counts.json file, sort it by count descending, and enumerate so position becomes rank: {word: rank} (rank 0 = most frequent).

    Args:
        counts_path: path to the json file containing the counts.
        keep_top_k: if not None, only extract the top keep_top_k words from the sorted rank map. This is useful if we know the ranking threshold K, as everything path K collapses to OOV_rank anyway.

    Returns:
        rank_map and OOV_rank (the length of the rank_map)
    """
    with open(counts_path) as f:
        counts = json.load(f)
    words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if keep_top_k is not None:
        words = words[:keep_top_k]
    rank_map = {w: r for r, (w, _) in enumerate(words)}
    return rank_map, len(rank_map)  # OOV_RANK = len(rank_map): always the rarest

def load_model(model_path: str, device):
    # Load model
    ckpt = torch.load(model_path, map_location='cuda')

    if 'state_dict' in ckpt: # Lightning save
        state_dict = ckpt['state_dict']
        # Strip the "model." Lightning prefix
        prefix = "model."
        stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    else:
        state_dict = ckpt 

    model = ModelWrapper("roberta-large").to(device)
    model.load_state_dict(stripped, strict=True)
    
    return model

def get_valid_authors(labels: torch.Tensor, min_texts_per_author: int) -> torch.Tensor:
    """Return a tensor of author IDs that have at least the minimum required texts."""
    counts = torch.bincount(labels)
    return torch.where(counts >= min_texts_per_author)[0]

def sample_author_chunks(labels: torch.Tensor, author_id: int, num_samples: int) -> torch.Tensor:
    """Sample a specified number of texts from a specific author."""
    auth_indices = torch.where(labels == author_id)[0]
    shuffled = torch.randperm(auth_indices.numel(), device=labels.device)
    return auth_indices[shuffled[:num_samples]]

@torch.no_grad()
def generate_embeddings(
    texts: list, 
    model, 
    tokenizer,
    device, 
    batch_size: int=64,
    ):
    """Generate embeddings for a given dataset.
    
    Args:
        texts: the text to turn into embeddings.
        model: the model returned by utils.load_model().
        tokenizer: tokenizer instance with added special tokens (<u> and <h>).
        device: cpu or cuda.
        batch_size: batch size.

    Returns:
        X: tensor of normalised embeddings.
    """
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)

        embeddings = model(input_ids, mask)
        all_embeddings.append(embeddings)

    X = torch.cat(all_embeddings, dim=0)

    return F.normalize(X, p=2, dim=-1)

class DummyModel(nn.Module):
    """Dummy model class for testing code. Simply returns the input ids."""
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_ids, mask, **kwargs):
        return input_ids.float()