"""Attention based models."""

from abc import ABC, abstractmethod
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def generate_causal_mask(T_seq: int, T_cached: int) -> Array:
    """Generates a causal mask for the sequence.

    Example: 4 cached tokens, 3 new tokens -> mask is 3x7:
    | 0 0 0 0 0 1 1 | (we mask out the last 2 tokens)
    | 0 0 0 0 0 0 1 | (we mask out the last token)
    | 0 0 0 0 0 0 0 | (we don't mask anything)

    Args:
        T_seq: The incoming sequence length.
        T_cached: The cached sequence length.

    Returns:
        A shape (T_seq, T_cached + T_seq) mask with ones representing masked tokens.
    """
    return jnp.concatenate((jnp.zeros((T_seq, T_cached)), jnp.triu(jnp.ones((T_seq, T_seq)), k=1)), axis=1)


def sinusoidal_pos_emb(seq_len: int, d_model: int, offset: int = 0) -> Array:
    """Computes sinusoidal positional encodings.

    Args:
        seq_len: Number of positions to encode.
        d_model: Dimensionality of the model (should match the embedding size).
        offset: Starting position index (useful during generation).

    Returns:
        An array of shape (seq_len, d_model) containing the positional encodings.
    """
    pos = jnp.arange(offset, offset + seq_len)[:, None]  # shape (seq_len, 1)
    i = jnp.arange(d_model)[None, :]  # shape (1, d_model)
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    angle_rads = pos * angle_rates
    # apply sin to even indices; cos to odd indices
    pos_encoding = jnp.where(i % 2 == 0, jnp.sin(angle_rads), jnp.cos(angle_rads))
    return pos_encoding


def get_rotary_emb(seq_len: int, head_dim: int, offset: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes rotary embeddings.

    Args:
        seq_len: The length of the sequence.
        head_dim: The dimension of the head.
        offset: The offset of the sequence.

    Returns:
        The cosine and sine of the rotary embeddings.
    """
    chex.assert_equal(head_dim % 2, 0)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim)) # [head_dim // 2]
    positions = jnp.arange(offset, offset + seq_len) # [seq_len]
    sinusoid_inp = positions @ inv_freq.T # [seq_len, head_dim // 2]
    cos = jnp.cos(sinusoid_inp)
    sin = jnp.sin(sinusoid_inp)
    # Interleave to create full head_dim vectors
    cos = jnp.reshape(jnp.stack([cos, cos], axis=-1), (seq_len, head_dim)) # [seq_len, head_dim]
    sin = jnp.reshape(jnp.stack([sin, sin], axis=-1), (seq_len, head_dim)) # [seq_len, head_dim]
    return cos, sin


def _rotate_every_two(x: jnp.ndarray) -> jnp.ndarray:
    """Rotates every two elements in the last dimension of the array."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rotated = jnp.stack((-x2, x1), axis=-1)
    return x_rotated.reshape(x.shape)


def apply_rotary(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies rotary embeddings to the query and key.

    Args:
        q: The query.
        k: The key.
        cos: The cosine of the rotary embeddings.
        sin: The sine of the rotary embeddings.

    Returns:
        The rotated query and key.
    """
    chex.assert_equal(q.shape, k.shape)
    chex.assert_equal(q.shape[-1] % 2, 0)
    chex.assert_equal(cos.shape, sin.shape)
    chex.assert_equal(cos.shape[-1], q.shape[-1])

    k_rot = (k * cos) + (_rotate_every_two(k) * sin)
    q_rot = (q * cos) + (_rotate_every_two(q) * sin)
    return q_rot, k_rot


class MultiHeadAttention(eqx.Module):
    dim: int = eqx.static_field()
    num_heads: int = eqx.static_field()
    head_dim: int = eqx.static_field()
    w_qkv: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.w_qkv = eqx.nn.Linear(dim, dim * 3, key=key)

    def apply_mask(self, attn_scores: Array, mask: Array) -> Array:
        T_q, T_kv = mask.shape
        chex.assert_shape(attn_scores, (..., T_q, T_kv))

        return jax.tree.map(lambda x, m: jnp.where(m, -jnp.inf, x), attn_scores, mask)

    def apply_attention(self, q: Array, k: Array, v: Array, mask: Array) -> Array:
        T_q, D = q.shape
        T_kv, D = k.shape
        chex.assert_shape(v, (T_kv, D))
        chex.assert_shape(mask, (T_q, T_kv))

        q = q.reshape(T_q, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, T_q, D_h]
        k = k.reshape(T_kv, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, T_kv, D_h]
        v = v.reshape(T_kv, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, T_kv, D_h]

        attn_scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(self.head_dim)  # [H, T_q, T_kv]
        attn_scores_masked = self.apply_mask(attn_scores, mask)
        attn_weights = jax.nn.softmax(attn_scores_masked, axis=-1)

        res = attn_weights @ v  # [H, T_q, D_h]
        return res.transpose(1, 0, 2).reshape(T_q, D)  # [T_q, D]

    def forward_sequence(self, x_seq: Array, kv_cache: tuple[Array, Array] | None) -> tuple[Array, tuple[Array, Array]]:
        T_seq = x_seq.shape[0]
        T_cached = kv_cache[0].shape[0] if kv_cache is not None else 0
        qkv_seq = jax.vmap(self.w_qkv)(x_seq)  # [T_seq, 3D]
        q_seq, k_seq, v_seq = jnp.split(qkv_seq, 3, axis=-1)

        if kv_cache is None:
            k = k_seq
            v = v_seq
        else:
            k = jnp.concatenate([kv_cache[0], k_seq], axis=0)  # [T_cached + T_seq, D]
            v = jnp.concatenate([kv_cache[1], v_seq], axis=0)

        mask = generate_causal_mask(T_seq, T_cached)  # [T_seq, T_cached + T_seq]
        attn_output = self.apply_attention(q_seq, k, v, mask)  # [T_seq, D]

        res = x_seq + attn_output  # [T_seq, D]
        return res, (k, v)


class RotaryMultiHeadAttention(MultiHeadAttention):
    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        super().__init__(dim, num_heads, key=key)

    def forward_sequence(
        self, x_seq: jnp.ndarray, kv_cache: tuple[jnp.ndarray, jnp.ndarray] | None
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        T_seq = x_seq.shape[0]
        T_cached = kv_cache[0].shape[0] if kv_cache is not None else 0

        # Compute q, k, v from the input sequence
        qkv_seq = jax.vmap(self.w_qkv)(x_seq)  # [T_seq, 3 * D]
        q_seq, k_seq, v_seq = jnp.split(qkv_seq, 3, axis=-1)

        # Reshape q_seq and k_seq to [T_seq, num_heads, head_dim]
        q_seq = q_seq.reshape(T_seq, self.num_heads, self.head_dim)
        k_seq = k_seq.reshape(T_seq, self.num_heads, self.head_dim)

        # Compute rotary embeddings for the new tokens only
        cos, sin = get_rotary_emb(T_seq, self.head_dim, offset=T_cached)
        q_seq, k_seq = apply_rotary(q_seq, k_seq, cos, sin)

        # Flatten back to [T_seq, D]
        q_seq = q_seq.reshape(T_seq, self.num_heads * self.head_dim)
        k_seq = k_seq.reshape(T_seq, self.num_heads * self.head_dim)

        if kv_cache is None:
            k = k_seq
            v = v_seq
        else:
            k = jnp.concatenate([kv_cache[0], k_seq], axis=0)
            v = jnp.concatenate([kv_cache[1], v_seq], axis=0)

        mask = generate_causal_mask(T_seq, T_cached)  # [T_seq, T_cached + T_seq]
        attn_output = self.apply_attention(q_seq, k, v, mask)  # [T_seq, D]
        res = x_seq + attn_output  # Residual connection
        return res, (k, v)


class AttentionBlock(eqx.Module):
    dim: int = eqx.static_field()
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: MultiHeadAttention | RotaryMultiHeadAttention
    mlp: eqx.nn.MLP

    def __init__(self, dim: int, num_heads: int, use_rotary_embeddings: bool, key: PRNGKeyArray):
        self.dim = dim

        k1, k2 = jax.random.split(key, 2)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)

        if use_rotary_embeddings:
            self.attn = RotaryMultiHeadAttention(dim, num_heads, key=k1)
        else:
            self.attn = MultiHeadAttention(dim, num_heads, key=k1)

        self.mlp = eqx.nn.MLP(dim, dim, width_size=4 * dim, depth=2, key=k2, activation=jax.nn.gelu)

    def forward_sequence(self, x_seq: Array, kv_cache: tuple[Array, Array] | None) -> tuple[Array, tuple[Array, Array]]:
        x_norm1 = jax.vmap(self.ln1)(x_seq)
        x_attn, kv_cache = self.attn.forward_sequence(x_norm1, kv_cache)
        x_res1 = x_seq + x_attn
        x_norm2 = jax.vmap(self.ln2)(x_res1)
        x_mlp = jax.vmap(self.mlp)(x_norm2)
        x_out = x_res1 + x_mlp
        return x_out, kv_cache


class Transformer(eqx.Module, ABC):
    hidden_size: int = eqx.static_field()
    max_context_length: int = eqx.static_field()
    vocab_embedding: eqx.nn.Embedding
    output_layer: eqx.nn.Linear
    blocks: list[AttentionBlock]

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        use_rotary_embeddings: bool,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.max_context_length = max_context_length
        keys = jax.random.split(key, num_layers + 2)
        self.vocab_embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=keys[0])
        self.blocks = [
            AttentionBlock(hidden_size, num_heads, use_rotary_embeddings, key=keys[i + 1]) for i in range(num_layers)
        ]
        self.output_layer = eqx.nn.Linear(hidden_size, vocab_size, key=keys[-1])

    @abstractmethod
    def input_pos_embedding(self, seq_len: int, offset: int) -> Array: ...
    """Position embeddings added directly to the input sequence.

    Args:
        seq_len: The length of the sequence.
        offset: The offset of the sequence.

    Returns:
        The position embeddings.
    """

    def forward_sequence(
        self, x: Array, kv_caches: list[tuple[Array, Array]] | None
    ) -> tuple[Array, list[tuple[Array, Array]]]:
        new_kv_caches = []
        for i in range(len(self.blocks)):
            x, kv_cache = self.blocks[i].forward_sequence(x, kv_caches[i] if kv_caches is not None else None)
            new_kv_caches.append(kv_cache)
        return x, new_kv_caches

    def predict_sequence(self, x_seq: Array) -> Array:
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)
        x_seq += self.input_pos_embedding(x_seq.shape[0], offset=0)
        y, _ = self.forward_sequence(x_seq, None)
        return jax.vmap(self.output_layer)(y)

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        x_prompt = jax.vmap(self.vocab_embedding)(prompt_seq)
        x_prompt += self.input_pos_embedding(x_prompt.shape[0], offset=0)
        x_emb, kv_caches = self.forward_sequence(x_prompt, None)

        predicted_tokens = []
        predicted_token = jnp.argmax(self.output_layer(x_emb[-1]), axis=-1)
        predicted_tokens.append(predicted_token)

        for _ in range(max_len):
            current_pos = kv_caches[0][0].shape[0] if kv_caches is not None else x_emb.shape[0]
            token_emb = self.vocab_embedding(predicted_token)[None, ...]
            token_emb += self.input_pos_embedding(1, offset=current_pos)
            x_emb, kv_caches = self.forward_sequence(token_emb, kv_caches)
            predicted_token = jnp.argmax(self.output_layer(x_emb[-1]), axis=-1)
            predicted_tokens.append(predicted_token)

        return jnp.array(predicted_tokens)


class TransformerNoPE(Transformer):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        key: PRNGKeyArray,
    ):
        super().__init__(vocab_size, hidden_size, num_layers, num_heads, max_context_length, False, key)

    def input_pos_embedding(self, seq_len: int, offset: int) -> Array:
        return jnp.zeros((seq_len, self.hidden_size))


class TransformerSinusoidalPE(Transformer):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        key: PRNGKeyArray,
    ):
        super().__init__(vocab_size, hidden_size, num_layers, num_heads, max_context_length, False, key)

    def input_pos_embedding(self, seq_len: int, offset: int) -> Array:
        return sinusoidal_pos_emb(seq_len, self.hidden_size, offset)


class TransformerLearnedPE(Transformer):
    pe: eqx.nn.Embedding

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        key: PRNGKeyArray,
    ):
        super().__init__(vocab_size, hidden_size, num_layers, num_heads, max_context_length, False, key)
        self.pe = eqx.nn.Embedding(max_context_length, hidden_size, key=key)

    def input_pos_embedding(self, seq_len: int, offset: int) -> Array:
        return jax.vmap(self.pe)(jnp.arange(offset, offset + seq_len))

class TransformerRotaryPE(Transformer):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        key: PRNGKeyArray,
    ):
        super().__init__(vocab_size, hidden_size, num_layers, num_heads, max_context_length, True, key)

    def input_pos_embedding(self, seq_len: int, offset: int) -> jnp.ndarray:
        # In rotary embeddings, the position information is injected inside attention.
        return jnp.zeros((seq_len, self.hidden_size))