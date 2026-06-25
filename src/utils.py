import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from src.model import ModelWrapper

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
    rank_tensor: torch.Tensor=None,
    masking_threshold: int=None,
    batch_size: int=64,
    ):
    """Generate embeddings for a given dataset.
    
    Args:
        texts: the text to turn into embeddings.
        model: the model returned by utils.load_model().
        tokenizer: tokenizer instance with added special tokens (<u> and <h>).
        device: cpu or cuda.
        rank_tensor: tensor containing the rank of each token in the vocab.
        masking_threshold: any token whose rank is >= this threshold will be masked
        batch_size: batch size.

    Returns:
        X: tensor of normalised embeddings.
    """
    model.eval()

    if rank_tensor is not None:
        rank_tensor = rank_tensor.to(device)

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

        is_masking_enabled = (rank_tensor is not None and masking_threshold is not None)
        if is_masking_enabled:
            input_ids = apply_content_mask(
                input_ids,
                rank_tensor,
                masking_threshold,
            )

        embeddings = model(input_ids, mask)
        all_embeddings.append(embeddings)

    X = torch.cat(all_embeddings, dim=0)

    return F.normalize(X, p=2, dim=-1)

def build_siamese_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Concatenate embeddings to form a siamese pair for authorship verifcation.
    Concatenates: [A, B, abs(A - B), A * B]

    Args:
        a: Tensor of shape (d) or (bs, d)
        b: Tensor of shape (d) or (bs, d)

    Returns:
        torch.Tensor: combined feature tensor. Shape (4*d) or (bs, 4*d)
    """

    abs_diff = torch.abs(a-b)
    product = a * b
    return torch.cat([a, b, abs_diff, product], dim=-1)

class DummyModel(nn.Module):
    """Dummy model class for testing code. Simply returns the input ids."""
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_ids, mask, **kwargs):
        return input_ids.float()
    
def calculate_masking_threshold(counts_path: str, strategy: str, value: float) -> int:
    """
    Calculate the integer rank boundary under which tokens are kept unmasked. Tokens with rank >= returned_integer will be replaced with <mask>.
    This allows us to apply different strategies without recomputing ranks or frequencies, as the data_builder.py only use a rank integer.

    Args:
        counts_path (str): filepath to the counts.json file (word or token).
        strategy (str): 'top_k', 'percentile', or 'relative_frequency'.
        value (int): parameter for the specified strategy.

    Returns:
        integer rank boundary. 
    """
    if not Path(counts_path).exists():
        raise FileNotFoundError(f"Counts file not found at {counts_path}")
    
    if strategy == 'top_k':
        # Keep top K most frequent tokens unmasked
        return int(value)
    
    with open(counts_path, 'r', encoding='utf-8') as f:
        counts = json.load(f)

    sorted_counts = sorted(counts.values(), reverse=True) # Descending order
    total_unique_tokens = len(sorted_counts)

    if strategy == 'percentile':
        # Keep top % most frequent tokens unmasked
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Percentile must be a fraction between 0.0 and 1.0")
        return int(total_unique_tokens * value)
    
    elif strategy == 'relative_frequency':
        # Keep tokens unmasked only if their relative frequency >= value
        total_token_occurences = sum(sorted_counts)
        keep_count = 0
        for count in sorted_counts:
            rel_freq = count / total_token_occurences
            if rel_freq >= value:
                keep_count += 1
            else:
                break
        return keep_count
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
def create_rank_tensor(rank_map: dict, oov_rank: int, tokenizer, protected_rank: int=-1):
    """
    Convert token_id -> rank dictionary into a 1d lookup tensor for quick indexing.
    """

    vocab_size = len(tokenizer)
    arr = torch.full((vocab_size,), fill_value=oov_rank, dtype=torch.int32) # Init everything with OOV

    special_ids = {
        tokenizer.bos_token_id,  # <s>
        tokenizer.eos_token_id,  # </s>
        tokenizer.pad_token_id,  # <pad>
        tokenizer.unk_token_id,  # <unk>
    }

    # Special tokens get rank -1
    for special_id in special_ids:
        if special_id is not None:
            arr[special_id] = protected_rank

    for id, rank in rank_map.items():
        id = int(id)
        if id < vocab_size:
            arr[id] = rank

    return arr

def apply_content_mask(input_ids: torch.Tensor, rank_tensor: torch.Tensor, threshold: int, mask_token: int=50264):
    """
    Apply content masking to a tensor of input_ids, for use in the generate_embeddings() function.
    """
    token_ranks = rank_tensor[input_ids]
    mask = (token_ranks >= threshold) & (token_ranks > -1)
    input_ids[mask] = mask_token
    return input_ids