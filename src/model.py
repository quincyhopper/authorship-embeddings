from transformers import AutoModel
import torch.nn as nn
import torch

class ModelWrapper(nn.Module):
    """Wrapper for transformer + projection layer."""
    def __init__(self, model_code:str):
        """
        Args:
            model_code (str): name of huggingface model, e.g. 'roberta-large'.
        """
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_code)
        self.d_model = self.transformer.config.hidden_size       # 1024 for RoBERTa-Large
        self.projection = nn.Linear(self.d_model, self.d_model)

    def forward(self, input_ids, attention_mask):
        """Take minibatches of tokenised sequences and return their embeddings

        Args:
            input_ids: tokenised sequences: [Minibatch size (Mb), Sequence length (L)].
            attention_mask: [Mb, L]

        Returns:
            Tensor of embedded sequences: [Mb, Embedding dimension (D)]
        """

        # Transformer pass: [Mb, L, D]
        outputs = self.transformer(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state

        # Mean pooling: [Mb, D]
        pooled_output = self.mean_pooling(token_embeddings, attention_mask)

        # Projection layer: [Mb, D]
        projected_output = self.projection(pooled_output)

        return projected_output

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pool the embedded tokens.
        
        Args:
            token_embeddings: tokenised sequences [Mb, L, D]
            attention_mask: [Mb, L, D]

        Returns:
            Mean pooled embeddings [Mb, D]
        """

        # Expand attenntion mask so that it match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Multiply embeddings by padding to zero-out padding (and sum rows)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Count non-padding tokens per text (account for div by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
    
