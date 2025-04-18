"""State space models."""

from abc import ABC, abstractmethod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))


class BaseSSMBlock(eqx.Module, ABC):
    @abstractmethod
    def forward(self, h: Array, x: Array) -> Array: ...

    @abstractmethod
    def forward_sequence(self, x_seq: Array) -> Array: ...

    @abstractmethod
    def get_a_mat(self, x: Array) -> Array: ...

    @abstractmethod
    def get_b_mat(self, x: Array) -> Array: ...


class SSMBlock(BaseSSMBlock):
    a_mat: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        key_a, key_b = jax.random.split(key)
        self.a_mat = glorot(key_a, (hidden_size, hidden_size))
        self.b_mat = glorot(key_b, (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        return self.a_mat

    def get_b_mat(self, x: Array) -> Array:
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        a_mat = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        h = a_mat @ h + b_mat.T @ x
        return h

    def forward_sequence(self, x_seq: Array) -> Array:
        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        a_mat = self.get_a_mat(x_seq)
        h_0 = jnp.zeros(a_mat.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x_seq)
        return h_seq


class DiagSSMBlock(BaseSSMBlock):
    a_diag: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.a_diag = glorot(keys[0], (hidden_size,))
        self.b_mat = glorot(keys[1], (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        return self.a_diag

    def get_b_mat(self, x: Array) -> Array:
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        a_diag = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        h = a_diag * h + b_mat.T @ x
        return h

    def forward_sequence(self, x_seq: Array, *, use_conv: bool = True, recursive_kernel_calc: bool = False) -> Array:
        """Performas a potentially parallelized forward pass across time."""
        if use_conv:
            return self._forward_sequence_conv(x_seq, recursive_kernel_calc=recursive_kernel_calc)
        else:
            return self._forward_sequence_scan(x_seq)

    def _get_kernel(self, x_seq: Array, length: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        exponents = jnp.arange(length)
        a_diag = self.get_a_mat(x_seq)
        kernel = jnp.power(a_diag[:, None], exponents)  # (H, T)
        kernel = kernel[:, None, :]  # (H, 1, T)
        return kernel

    def _get_kernel_recursive(self, x_seq: Array, length: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        assert length % 2 == 0, "Length must be even."
        a_diag = self.get_a_mat(x_seq)

        def helper(length: int) -> tuple[Array, Array]:
            """Returns the kernel and the sqrt of the diagonal."""
            if length == 1:
                return jnp.ones_like(a_diag)[:, None], a_diag[:, None]

            half_length = length // 2
            kernel_half, a_half = helper(half_length)
            kernel = jnp.concatenate([kernel_half, a_half * kernel_half], axis=-1)
            return kernel, a_half * a_half

        kernel, a_diag = helper(length)
        return kernel[:, None, :]  # (H, 1, L)

    def _forward_sequence_conv(self, x_seq: Array, *, recursive_kernel_calc: bool = False) -> Array:
        """Convolves x (T, H) across time using the kernel."""
        seq_len, hidden_size = x_seq.shape
        b_mat = self.get_b_mat(x_seq)

        s = b_mat.T @ x_seq.T  # (H, T)
        s_padded = jnp.pad(s, ((0, 0), (seq_len - 1, 0)))[None, :, :]  # (1, H, 2T-1)

        if recursive_kernel_calc:
            kernel = self._get_kernel_recursive(x_seq, seq_len)
        else:
            kernel = self._get_kernel(x_seq, seq_len)

        kernel_flipped = jnp.flip(kernel, axis=-1)  # (H, 1, L)

        conv_out = jax.lax.conv_general_dilated(
            s_padded,
            kernel_flipped,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCT", "OIT", "NCT"),  # convolving over time
            feature_group_count=hidden_size,
        )
        conv_out = conv_out[0].T  # (T, H)
        return conv_out

    def _forward_sequence_scan(self, x_seq: Array) -> Array:
        """Naively forward across time."""

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        a_diag = self.get_a_mat(x_seq)
        h_0 = jnp.zeros(a_diag.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x_seq)
        return h_seq


class DiscreteDiagSSMBlock(DiagSSMBlock):
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        init_delta: float = 0.1,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__(hidden_size, key=key)
        self.delta = jnp.array(init_delta)
        self.a_diag = jax.random.uniform(key, (hidden_size,), minval=-1.0, maxval=0.0) * init_scale

    def get_a_mat(self, x: Array) -> Array:
        """Discretize the diagonal matrix using zero-order hold."""
        a_diag_discrete = jnp.exp(self.a_diag * self.delta)
        return a_diag_discrete

    def get_b_mat(self, x: Array) -> Array:
        """Discretize the input matrix using zero-order hold."""
        delta_a_diag = self.a_diag * self.delta
        exp_a_diag = jnp.exp(delta_a_diag)
        delta_a_inv = 1 / delta_a_diag
        delta_b_mat = self.delta * self.b_mat

        b_discrete = delta_a_inv * (exp_a_diag - 1) * delta_b_mat
        return b_discrete


class DPLRSSMBlock(BaseSSMBlock):
    a_diag: Array
    p_vec: Array
    q_vec: Array
    b_mat: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        rank: int = 1,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.a_diag = glorot(k1, (hidden_size,))
        self.p_vec = glorot(k2, (hidden_size, rank))
        self.q_vec = glorot(k3, (hidden_size, rank))
        self.b_mat = glorot(k4, (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        """Construct discretized A matrix: diag(a_diag) + P Q^T, exponentiated."""
        a_mat = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T
        # A_disc = jax.scipy.linalg.expm(self.delta * A)
        # return A_disc
        return a_mat

    def get_b_mat(self, x: Array) -> Array:
        """Discretize B using: ∫ exp(A τ) dτ B ≈ A^{-1}(exp(A Δ) - I) B"""
        # a_mat = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T
        # exp_a_mat = jax.scipy.linalg.expm(self.delta * a_mat)
        # a_mat_inv = jnp.linalg.pinv(a_mat)
        # b_mat_disc = a_mat_inv @ (exp_a_mat - jnp.eye(a_mat.shape[0])) @ self.b_mat
        # return b_mat_disc
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        A = self.get_a_mat(x)
        B = self.get_b_mat(x)
        return A @ h + B @ x

    def forward_sequence(self, x_seq: Array) -> Array:
        """Performs a forward pass across time."""

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        a_mat = self.get_a_mat(x_seq)
        h_0 = jnp.zeros(a_mat.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x_seq)
        return h_seq


class SSM(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    output_layer: eqx.nn.Linear
    blocks: list[BaseSSMBlock]
    num_layers: int = eqx.static_field()
    hidden_size: int = eqx.static_field()
    skip_connections: bool = eqx.static_field()

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        block_type: Literal["diagonal", "full_rank", "dplr"] = "full_rank",
        skip_connections: bool = False,
        discretize: bool = False,
        *,
        key: PRNGKeyArray,
        disc_init_delta: float = 0.1,
        disc_init_scale: float = 10.0,
        dplr_rank: int = 1,
    ) -> None:
        vocab_key, s4_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=vocab_key)
        self.output_layer = eqx.nn.Linear(hidden_size, vocab_size, key=key)

        block_keys = jax.random.split(s4_key, num_layers)

        def get_block(key: PRNGKeyArray) -> BaseSSMBlock:
            """Returns a block of the given type."""
            match block_type:
                case "diagonal":
                    return (
                        DiscreteDiagSSMBlock(
                            hidden_size, key=key, init_delta=disc_init_delta, init_scale=disc_init_scale
                        )
                        if discretize
                        else DiagSSMBlock(hidden_size, key=key)
                    )
                case "full_rank":
                    if discretize:
                        raise ValueError("Full rank blocks do not support discretization due to instability.")
                    return SSMBlock(hidden_size, key=key)
                case "dplr":
                    if discretize:
                        raise ValueError("DPLR blocks do not support discretization yet.")
                    return DPLRSSMBlock(hidden_size, rank=dplr_rank, key=key)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = [get_block(block_keys[i]) for i in range(num_layers)]
        self.skip_connections = skip_connections
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        """Performs a single forward pass."""
        new_hs = []
        for i, block in enumerate(self.blocks):
            h = block.forward(hs[i], x)
            new_hs.append(h)
            xh = jax.nn.gelu(h)
            x = xh + x if self.skip_connections else xh
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        """Predicts an entire sequence of tokens at once."""
        x_emb = jax.vmap(self.vocab_embedding)(x_seq)
        for block in self.blocks:
            h = block.forward_sequence(x_emb)
            # h = block.naive_forward_sequence(x_emb)
            h = jax.nn.gelu(h)
            x_emb = h + x_emb if self.skip_connections else h
        y = jax.vmap(self.output_layer)(x_emb)
        return y

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        """Autoregressively generates a sequence of tokens."""
        hs = [jnp.zeros(self.hidden_size) for _ in range(self.num_layers)]
        prompt_seq_embedded = jax.vmap(self.vocab_embedding)(prompt_seq)

        def encode_step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[Array], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[Array], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self.vocab_embedding(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.PRNGKey(0)), None, length=max_len)

        return sequence
