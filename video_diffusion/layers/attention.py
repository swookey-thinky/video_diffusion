"""Attention layers to use with DDPM.

This package implements multi-head self/cross attention from "Attention Is All You Need".
"""

from einops import rearrange
from functools import partial
import math
import torch
from torch.jit import Final
from torch.nn.init import xavier_uniform_
from typing import Optional, Dict

from video_diffusion.layers.utils import (
    conv_nd,
    zero_module,
    ContextBlock,
    EinopsToAndFrom,
)
from video_diffusion.context import (
    ContextAdapter,
    NullContextAdapter,
)
from video_diffusion.utils import instantiate_from_config

try:
    from xformers import ops as xops
except ImportError:
    xops = None
    print(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers\npip install xformers."
    )


class SpatialCrossAttention(ContextBlock):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.

    When the context_dim is None or -1, this is equivalent to Multi-Head Self Attention.
    And when heads=1 and dim_head=in_channels, this is equivalent to the self attention.

    The input to this block is of shape (B, C, *spatial), and the additional context
    comes in with shape (B, context_dim). This attention block is a good
    layer to use when the input data is spatial, and can be used for Multi-Head Attention,
    Multi-Head Self Attention, and Multi-Head Cross Attention.
    """

    def __init__(
        self,
        in_channels,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        pre_layer_norm: bool = False,
        post_layer_norm: bool = False,
        context_layer_norm: bool = False,
        context_adapter: Dict = {},
    ):
        """Initialize a new instance of SpatialCrossAttention."""
        super().__init__()

        if context_dim == -1:
            context_dim = None
        self._channels = in_channels
        if dim_head == -1:
            self._num_heads = heads
        else:
            assert (
                in_channels % dim_head == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {dim_head}"
            self._num_heads = in_channels // dim_head
        if pre_layer_norm:
            self._norm = ChanLayerNorm(in_channels)
        else:
            self._norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)
        if post_layer_norm:
            self._final_norm = ChanLayerNorm(in_channels)
        else:
            self._final_norm = torch.nn.Identity()

        if context_layer_norm:
            self._context_layer_norm = ChanLayerNorm(context_dim, dim=-2)
        else:
            self._context_layer_norm = torch.nn.Identity()

        self._qkv = conv_nd(1, in_channels, in_channels * 3, 1)
        self._attention = QKVAttention(self._num_heads)

        if "target" in context_adapter:
            self._context_adapter = instantiate_from_config(context_adapter)
        else:
            self._context_adapter = NullContextAdapter()

        if context_dim is not None:
            self._encoder_kv = conv_nd(1, context_dim, in_channels * 2, 1)
        self._proj_out = zero_module(conv_nd(1, in_channels, in_channels, 1))
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, x, context: Optional[Dict] = None):
        b, c, *spatial = x.shape
        qkv = self._qkv(self._norm(x).view(b, c, -1))
        if context is not None:
            context = self._context_layer_norm(self._context_adapter(context))
            if context is not None:
                encoder_out = self._encoder_kv(context)
            else:
                encoder_out = None
            h = self._attention(qkv, encoder_out)
        else:
            h = self._attention(qkv)
        h = self._proj_out(h)
        h = h.reshape(b, c, *spatial)
        h = self._final_norm(h)
        return x + self._dropout(h)


class QKVAttention(torch.nn.Module):
    """A module which performs QKV attention."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv, encoder_kv=None):
        """Apply QKV attention.

        Args:
            qkv: an [B x (H * C * 3) x T] tensor of Qs, Ks, and Vs.

        Returns:
            A [B x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, length).split(ch, dim=1)

        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.num_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.num_heads, ch * 2, -1).split(
                ch, dim=1
            )
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class MultiHeadSelfAttentionWithRelativePosition(ContextBlock):
    """Self Attention block with relative position embeddings.

    Accepts tensor input of size (B, F, C), where F is the number of
    frames in the video, and C is the number of channels.
    """

    def __init__(
        self,
        in_channels: int,
        dim_attention_head: int,
        dropout: float = 0.0,
        context_adapter: Dict = {},
        **kwargs,
    ):
        super().__init__()

        num_attention_heads = in_channels // dim_attention_head
        hidden_size = num_attention_heads * dim_attention_head
        self.head_size = dim_attention_head
        self.num_heads = num_attention_heads

        self.linear_q = torch.nn.Linear(in_channels, hidden_size)
        self.linear_k = torch.nn.Linear(in_channels, hidden_size)
        self.linear_v = torch.nn.Linear(in_channels, hidden_size)
        self._norm = EinopsToAndFrom(
            from_einops="b f c",
            to_einops="b c f",
            fn=torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
        )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear_out = torch.nn.Linear(hidden_size, in_channels)

        self.linear_pos = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_bias_u = torch.nn.Parameter(
            torch.zeros(self.num_heads, self.head_size)
        )
        self.pos_bias_v = torch.nn.Parameter(
            torch.zeros(self.num_heads, self.head_size)
        )

        if "target" in context_adapter:
            self._context_adapter = instantiate_from_config(context_adapter)
        else:
            self._context_adapter = NullContextAdapter()

        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        context: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape `(B, L, D)`
            position_embeddings (torch.Tensor): with shape `(B, L, D)`
            attention_mask (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape`(B, L, D)`
        """
        assert "relative_position_embedding" in context
        batch_size, sequence_length, hidden_size = x.size()

        x_norm = self._norm(x)

        # `(B, L, D)` -> `(B, L, H, D/H)`
        query = self.linear_q(x_norm).view(
            batch_size, -1, self.num_heads, self.head_size
        )
        key = self.linear_k(x_norm).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(x_norm).view(
            batch_size, -1, self.num_heads, self.head_size
        )

        # `(B, L, H, D/H)` -> `(B, L, H, D/H)`
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        position_embeddings = context["relative_position_embedding"]
        scores = self._apply_relative_embeddings(
            query=query, key=key, relative_position_embeddings=position_embeddings
        )

        attention_mask = None
        if "attention_mask" in context:
            attention_mask = context["attention_mask"]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(
                attention_mask == 0, torch.finfo(scores.dtype).min
            )

        probs = torch.softmax(scores, dim=-1)

        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_size
        )
        out = self.linear_out(hidden_states)

        return x + self.dropout(out)

    def _apply_relative_embeddings(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        relative_position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate attention weight with relative position by Skew algorithm.

        Args:
            query (torch.Tensor): with shape `(B, H, L, D/H)`
            key: (torch.Tensor): with shape `(B, H, L, D/H)`
            relative_position_embeddings (torch.Tensor): with shape `(L, L, D)`

        Returns:
            torch.Tensor with shape `(B, H, L, L)`

        """

        # `(L, L, D)` -> `(H, L, L, D/H)`
        proj_relative_position_embeddings = self.linear_pos(
            relative_position_embeddings
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(
            1, 2
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(
            0, 1
        )

        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        scores_bd = (
            q_with_bias_v.unsqueeze(2) * proj_relative_position_embeddings.unsqueeze(0)
        ).sum(-1)

        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_size: int, max_source_positions: int):
        super().__init__()
        self.positional_params = torch.nn.Parameter(
            torch.randn(max_source_positions * 2 - 1, hidden_size)
        )
        self.max_length = max_source_positions

    def forward(self, x: torch.Tensor, **kwargs):
        input_length = x.size(1)
        position_ids = torch.arange(input_length)
        relative_position_matrix = position_ids[None, :] - position_ids[:, None]
        relative_position_matrix = relative_position_matrix + self.max_length - 1

        relative_position_embeddings = self.positional_params[relative_position_matrix]

        return relative_position_embeddings


class AttentionPooling(torch.nn.Module):

    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x):
        bs, length, width = x.size()

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(
            x.dtype
        )
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = torch.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token


class LayerNorm(torch.nn.Module):
    def __init__(self, feats, stable=False, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = torch.nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim
        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


ChanLayerNorm = partial(LayerNorm, dim=-3)


class MultiHeadSelfAttention(torch.nn.Module):
    """Multi-Head Self Attention class used by DiT.

    Based on the transformer attention in vision transformers from here:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L58C1-L106C17
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_fused_attn: bool = False,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            assert hasattr(
                torch.nn.functional, "scaled_dot_product_attention"
            ), "Torch version does not have scaled_dot_product_attention. disable fused attention."
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class QKVAttentionWithRelativePosition(torch.nn.Module):
#     """A module which performs QKV attention using relative positions.

#     Based on the TF implementation here:
#     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L1739
#     """

#     def __init__(self, num_heads: int, max_relative_position: int, depth: int):
#         super().__init__()
#         # Depth is the dimension per head
#         self.num_heads = num_heads
#         vocab_size = max_relative_position * 2 + 1

#         # Generates embedding for each relative position of dimension depth.
#         self._k_embeddings_table = torch.nn.Parameter(torch.zeros(vocab_size, depth))
#         self._v_embeddings_table = torch.nn.Parameter(torch.zeros(vocab_size, depth))

#     def forward(self, qkv, encoder_kv=None):
#         """Apply QKV attention.

#         Args:
#             qkv: an [B x (H * C * 3) x T] tensor of Qs, Ks, and Vs.

#         Returns:
#             A [B x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.num_heads) == 0
#         ch = width // (3 * self.num_heads)
#         q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, length).split(ch, dim=1)

#         if encoder_kv is not None:
#             assert encoder_kv.shape[1] == self.num_heads * ch * 2
#             ek, ev = encoder_kv.reshape(bs * self.num_heads, ch * 2, -1).split(
#                 ch, dim=1
#             )
#             k = torch.cat([ek, k], dim=-1)
#             v = torch.cat([ev, v], dim=-1)

#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = torch.einsum(
#             "bct,bcs->bts", q * scale, k * scale
#         )  # More stable with f16 than dividing afterwards
#         weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = torch.einsum("bts,bcs->bct", weight, v)
#         return a.reshape(bs, -1, length)


# def dot_product_attention_relative(
#     q,
#     k,
#     v,
#     bias,
#     max_relative_position,
#     dropout_rate=0.0,
#     image_shapes=None,
#     hard_attention_k=0,
#     gumbel_noise_weight=0.0,
# ):
#     """Calculate relative position-aware dot-product self-attention.

#     The attention calculation is augmented with learned representations for the
#     relative position between each element in q and each element in k and v.

#     Args:
#         q: a Tensor with shape [batch, heads, length, depth].
#         k: a Tensor with shape [batch, heads, length, depth].
#         v: a Tensor with shape [batch, heads, length, depth].
#         bias: bias Tensor.
#         max_relative_position: an integer specifying the maximum distance between
#             inputs that unique position embeddings should be learned for.
#         dropout_rate: a floating point number.
#         image_shapes: optional tuple of integer scalars.
#         save_weights_to: an optional dictionary to capture attention weights
#         for visualization; the weights tensor will be appended there under
#         a string key created from the variable scope (including name).
#         name: an optional string.
#         make_image_summary: Whether to make an attention image summary.
#         cache: whether use cache mode
#         allow_memory: whether to assume that recurrent memory is in use. If True,
#         the length dimension of k/v/bias may be longer than the queries, and it is
#         assumed that the extra memory entries precede the non-memory entries.
#         hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
#         gumbel_noise_weight: if > 0, apply Gumbel noise with weight
#         `gumbel_noise_weight` before picking top-k. This is a no op if
#         hard_attention_k <= 0.

#     Returns:
#         A Tensor.

#     Raises:
#         ValueError: if max_relative_position is not > 0.
#     """
#     if not max_relative_position:
#         raise ValueError(
#             "Max relative position (%s) should be > 0 when using "
#             "relative self attention." % (max_relative_position)
#         )

#     # Use separate embeddings suitable for keys and values.
#     depth = k.shape[3]
#     length_k = k.shape[2]
#     length_q = length_k

#     relations_keys = _generate_relative_positions_embeddings(
#         length_q,
#         length_k,
#         depth,
#         max_relative_position,
#     )
#     relations_values = _generate_relative_positions_embeddings(
#         length_q,
#         length_k,
#         depth,
#         max_relative_position,
#     )

#     # Compute self attention considering the relative position embeddings.
#     logits = _relative_attention_inner(q, k, relations_keys, True)
#     if bias is not None:
#         logits += bias
#     weights = torch.nn.functional.softmax(logits)
#     if hard_attention_k > 0:
#         weights = harden_attention_weights(
#             weights, hard_attention_k, gumbel_noise_weight
#         )
#     # weights = torch.nn.functional.dropout(weights, 1.0 - dropout_rate)
#     return _relative_attention_inner(weights, v, relations_values, False)


# def _generate_relative_positions_embeddings(
#     length_q, length_k, depth, max_relative_position
# ):
#     """Generates tensor of size [1 if cache else length_q, length_k, depth]."""
#     relative_positions_matrix = _generate_relative_positions_matrix(
#         length_q, length_k, max_relative_position
#     )
#     vocab_size = max_relative_position * 2 + 1
#     # Generates embedding for each relative position of dimension depth.
#     embeddings_table = torch.nn.Parameter(torch.zeros(vocab_size, depth))
#     embeddings = torch.gather(embeddings_table, dim=1, index=relative_positions_matrix)
#     return embeddings


# def _generate_relative_positions_matrix(length_q, length_k, max_relative_position):
#     """Generates matrix of relative positions between inputs."""
#     if length_q == length_k:
#         range_vec_q = range_vec_k = torch.range(0, length_q)
#     else:
#         range_vec_k = torch.range(0, length_k)
#         range_vec_q = range_vec_k[-length_q:]
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#     distance_mat_clipped = torch.clamp(
#         distance_mat, -max_relative_position, max_relative_position
#     )
#     # Shift values to be >= 0. Each integer still uniquely identifies a relative
#     # position difference.
#     final_mat = distance_mat_clipped + max_relative_position
#     return final_mat


# def _relative_attention_inner(x, y, z, transpose):
#     """Relative position-aware dot-product attention inner calculation.

#     This batches matrix multiply calculations to avoid unnecessary broadcasting.

#     Args:
#         x: Tensor with shape [batch_size, heads, length or 1, length or depth].
#         y: Tensor with shape [batch_size, heads, length or 1, depth].
#         z: Tensor with shape [length or 1, length, depth].
#         transpose: Whether to transpose inner matrices of y and z. Should be true if
#             last dimension of x is depth, not length.

#     Returns:
#         A Tensor with shape [batch_size, heads, length, length or depth].
#     """
#     batch_size, heads, length = x.shape
#     # batch_size = tf.shape(x)[0]
#     # heads = x.get_shape().as_list()[1]
#     # length = tf.shape(x)[2]

#     # xy_matmul is [batch_size, heads, length or 1, length or depth]
#     xy_matmul = torch.matmul(x, y.transpose() if transpose else y)
#     # x_t is [length or 1, batch_size, heads, length or depth]
#     x_t = torch.permute(x, (2, 0, 1, 3))
#     # x_t_r is [length or 1, batch_size * heads, length or depth]
#     x_t_r = torch.reshape(x_t, (length, heads * batch_size, -1))
#     # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
#     x_tz_matmul = torch.matmul(x_t_r, z.transpose() if transpose else z)
#     # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
#     x_tz_matmul_r = torch.reshape(x_tz_matmul, (length, batch_size, heads, -1))
#     # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
#     x_tz_matmul_r_t = torch.permute(x_tz_matmul_r, (1, 2, 0, 3))
#     return xy_matmul + x_tz_matmul_r_t


# def harden_attention_weights(weights, k, gumbel_noise_weight):
#     """Make attention weights non-0 only on the top k ones."""
#     if gumbel_noise_weight > 0.0:
#         gumbel_noise = -torch.log(
#             -torch.log(torch.clamp(torch.randn_like(weights), min=1e-5, max=1 - 1e-5))
#         )

#         weights += gumbel_noise * gumbel_noise_weight

#     # Subtract the top-kth weight and zero-out all lower ones.
#     # Note that currently in case of numerical ties it will retain more
#     # than k elements. In the future, we may want to avoid this.
#     weights -= common_layers.top_kth_iterative(weights, k)
#     weights = torch.nn.functional.relu(weights)
#     # Re-normalize the weights.
#     weights_sum = torch.sum(weights, dim=-1, keepdim=True)
#     weights_sum = torch.maximum(weights_sum, 1e-6)  # Avoid division by 0.
#     weights /= weights_sum
#     return weights
