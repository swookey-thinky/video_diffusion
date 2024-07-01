from abc import abstractmethod
from einops.layers.torch import Rearrange
import math
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union

from video_diffusion.utils import freeze, prob_mask_like
from video_diffusion.layers.attention import (
    AttentionPooling,
    RelativePositionalEncoding,
)
from video_diffusion.layers.utils import ContextBlock, Format, nchw_to, to_2tuple

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


class ContextEmbedSequential(torch.nn.Sequential):
    """Sequential module for timestep and conditional embeddings.

    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, context):
        for layer in self:
            if isinstance(layer, ContextBlock):
                x = layer(x, context=context)
            else:
                x = layer(x)
        return x


class SinusoidalPositionEmbedding(torch.nn.Module):
    """Implementation of Sinusoidal Position Embedding.

    Originally introduced in the paper "Attention Is All You Need",
    the original tensorflow implementation is here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L408
    """

    def __init__(self, embedding_dim, theta=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta

    def forward(self, x, **kwargs):
        device = x.device
        half_dim = self.embedding_dim // 2
        embedding = math.log(self.theta) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


class TimestepEmbeddingProjection(torch.nn.Module):
    def __init__(self, num_features: int, time_embedding_mult: int):
        super().__init__()
        time_embedding_dimension = num_features * time_embedding_mult
        self._projection = torch.nn.Sequential(
            SinusoidalPositionEmbedding(num_features),
            torch.nn.Linear(num_features, time_embedding_dimension),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dimension, time_embedding_dimension),
        )

    def forward(self, timestep: torch.Tensor, **kwargs):
        # Make sure there are no NaNs in the timestep embedding.
        # This is a debugging step because on my local 3090
        # this seems to happen sometimes, not sure why.
        projection = self._projection(timestep)
        if torch.isnan(projection).any():
            print(timestep)
            print(projection)
            assert False
        return projection


class RelativePositionEmbedding(torch.nn.Module):
    def __init__(self, hidden_size: int, max_source_positions: int, **kwargs):
        super().__init__()
        self._relative_positional_encoding = RelativePositionalEncoding(
            hidden_size=hidden_size, max_source_positions=max_source_positions
        )

    def forward(self, x: torch.Tensor, **kwargs):
        return self._relative_positional_encoding(x)


class PooledTextEmbeddingsToTimestep(torch.nn.Module):
    def __init__(
        self,
        text_embedding_dim: int,
        time_embedding_dim: int,
        attention_pooling_heads: int,
        **kwargs,
    ):
        super().__init__()
        self._encoder_pooling = torch.nn.Sequential(
            torch.nn.LayerNorm(text_embedding_dim),
            AttentionPooling(attention_pooling_heads, text_embedding_dim),
            torch.nn.Linear(text_embedding_dim, time_embedding_dim),
            torch.nn.LayerNorm(time_embedding_dim),
        )

    def forward(self, context: Dict):
        assert "text_embeddings" in context
        assert "timestep_embedding" in context
        pooling_out = self._encoder_pooling(context["text_embeddings"])
        timestep_embedding = context["timestep_embedding"]
        timestep_embedding = timestep_embedding + pooling_out.to(timestep_embedding)
        context["timestep_embedding"] = timestep_embedding
        return context


class RunProjection(torch.nn.Module):
    """Runs a defined projection."""

    def __init__(
        self,
        input_context_key: str,
        output_context_key: str,
        projection_key: str,
        projections: torch.nn.ModuleDict,
        **kwargs,
    ):
        super().__init__()
        self._input_context_key = input_context_key
        self._output_context_key = output_context_key
        self._projection_key = projection_key
        self._projections = projections

    def forward(self, context: Dict, device, **kwargs):
        assert (
            self._input_context_key in context
        ), f"{self._input_context_key} not found for projection {self._projection_key}."
        assert self._projection_key in self._projections

        context[self._output_context_key] = self._projections[self._projection_key](
            context[self._input_context_key], context=context, device=device
        )
        return context
