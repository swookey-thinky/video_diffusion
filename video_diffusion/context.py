"""Class for holding model conditioning.

Incorporates different conditioning available to the model, such
as timesteps, class labels, image embeddings, text embeddings, etc.

Possible conditioning signals include:

classes
timestep
timestep_embedding
text_tokens
text_embeddings
image_imbeddings
"""

from abc import abstractmethod
import torch
from transformers import T5Tokenizer
from typing import Dict, List


class ContextAdapter(torch.nn.Module):
    """Basic block which accepts a context conditioning."""

    @abstractmethod
    def forward(self, context: Dict):
        """Apply the module to `x` given `context` conditioning."""


class NullContextAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict):
        return None


class IgnoreContextAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict, *args, **kwargs):
        return context


class IgnoreInputPreprocessor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
