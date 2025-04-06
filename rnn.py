"""Recurrent neural networks."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class RNN(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.GRUCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, num_layers)
        self.rnn_cells = [
            eqx.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size, key=keys[i]) for i in range(num_layers)
        ]
        self.output_layer = eqx.nn.Linear(hidden_size, vocab_size, key=keys[-1])

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        """Performs a single forward pass."""
        new_hs = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            h = rnn_cell(x, hs[i])
            new_hs.append(h)
            x = h  # Pass the output of the current layer as input to the next
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        """Predicts an entire sequence of tokens at once."""
        hs = [jnp.zeros(cell.hidden_size) for cell in self.rnn_cells]
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)

        def step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_seq)
        return y_seq

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        """Autoregressively generates a sequence of tokens."""
        hs = [jnp.zeros(cell.hidden_size) for cell in self.rnn_cells]
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


class LSTM(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.LSTMCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(vocab_size, hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, num_layers)
        self.rnn_cells = [
            eqx.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, key=keys[i]) for i in range(num_layers)
        ]
        self.output_layer = eqx.nn.Linear(hidden_size, vocab_size, key=keys[-1])

    def __call__(self, hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
        """Performs a single forward pass."""
        new_hs: list[tuple[Array, Array]] = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            h, c = rnn_cell(x, hs[i])
            new_hs.append((h, c))
            x = h  # Pass the output of the current layer as input to the next
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        """Predicts an entire sequence of tokens at once."""
        hs = [(jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) for cell in self.rnn_cells]
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)

        def step(hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_seq)
        return y_seq

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        """Autoregressively generates a sequence of tokens."""
        hs = [(jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) for cell in self.rnn_cells]
        prompt_seq_embedded = jax.vmap(self.vocab_embedding)(prompt_seq)

        def encode_step(hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[tuple[Array, Array]], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[tuple[Array, Array]], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self.vocab_embedding(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.PRNGKey(0)), None, length=max_len)

        return sequence