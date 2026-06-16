import json
import torch
from model import ModelWrapper

def load_rank_map(counts_path: str, keep_top_k: int | None = None) -> tuple[dict, int]:
    """Load the word_counts.json file, sort it by count descending, and enumerate so position becomes rank: {word: rank} (rank 0 = most frequent).

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