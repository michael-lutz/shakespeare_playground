"""Core training code for Shakespeare character-level language models."""

from dataclasses import dataclass
from typing import Iterator, Literal, Protocol, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from jaxtyping import Array, PRNGKeyArray, PyTree

from attention import (
    TransformerLearnedPE,
    TransformerNoPE,
    TransformerRotaryPE,
    TransformerSinusoidalPE,
)
from rnn import LSTM, RNN
from ssm import SSM
import xax


@dataclass(frozen=True)
class ShakespeareDataset:
    text: str
    vocab: list[str]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


def load_shakespeare_text() -> ShakespeareDataset:
    """Loads the Tiny Shakespeare dataset.

    This function loads the tiny_shakespeare dataset from tfds, extracts the
    text, and builds a character-level tokenizer.

    Returns:
        The loaded dataset.
    """
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)

    # tiny_shakespeare consists of a single example with the full text.
    for example in tfds.as_numpy(ds):
        # the text field might be bytes, so decode if necessary.
        text = example["text"]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        break

    # Build vocabulary from unique characters in the text.
    vocab = sorted(list(set(text)))
    token_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_token = {i: ch for i, ch in enumerate(vocab)}
    return ShakespeareDataset(text, vocab, token_to_id, id_to_token)


@dataclass
class Config(xax.Config):
    vocab_size: int = xax.field(65)
    num_layers: int = xax.field(1)
    hidden_size: int = xax.field(256)
    batch_size: int = xax.field(12)
    learning_rate: float = xax.field(1e-3)
    max_context_length: int = xax.field(100)
    valid_every_n_seconds: float = xax.field(30.0)
    model_type: str = xax.field("lstm", help="The model to use")
    # Transformer-specific fields
    position_embedding_type: str = xax.field("rope", help="The type of transformer to use (if applicable)")
    use_start_token: bool = xax.field(True, help="Whether to use a start token (if applicable)")
    num_heads: int = xax.field(1, help="The number of attention heads to use (if applicable)")
    # SSM-specific fields
    ssm_type: str = xax.field("full_rank", help="The type of SSM block to use")
    ssm_discretize: bool = xax.field(False, help="Whether to discretize the SSM blocks")
    ssm_dplr_rank: int = xax.field(1, help="The rank of the DPLR blocks")


class SequenceModel(Protocol):
    def predict_sequence(self, x_seq: Array) -> Array: ...

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array: ...


class ShakespearePrediction(xax.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.ds = load_shakespeare_text()
        self.token_ids = jnp.array([self.ds.token_to_id[c] for c in self.ds.text], dtype=jnp.int32)

    def compute_metrics(
        self,
        model: PyTree,
        batch: tuple[Array, Array],
        output: Array,
        loss: Array,
        state: xax.State,
    ) -> dict[str, Array]:
        _, y = batch
        yhat = output.argmax(axis=-1)
        return {
            "loss": loss,
            "acc": (yhat == y).astype(float).mean(),
        }

    def get_model(self, key: PRNGKeyArray) -> SequenceModel:
        match self.config.model_type:
            case "rnn":
                return RNN(
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    key=key,
                )
            case "lstm":
                return LSTM(
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    key=key,
                )
            case "ssm":
                valid_types = get_args(Literal["diagonal", "full_rank", "dplr"])
                if self.config.ssm_type not in valid_types:
                    raise ValueError(f"Invalid ssm_type: {self.config.ssm_type}. Must be one of {valid_types}")

                block_type = cast(Literal["diagonal", "full_rank", "dplr"], self.config.ssm_type)
                return SSM(
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    block_type=block_type,
                    skip_connections=True,
                    discretize=self.config.ssm_discretize,
                    dplr_rank=self.config.ssm_dplr_rank,
                    key=key,
                )
            case "transformer":
                match self.config.position_embedding_type:
                    case "rope":
                        return TransformerRotaryPE(
                            vocab_size=self.config.vocab_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.num_layers,
                            num_heads=self.config.num_heads,
                            max_context_length=self.config.max_context_length,
                            use_start_token=self.config.use_start_token,
                            key=key,
                        )
                    case "nope":
                        return TransformerNoPE(
                            vocab_size=self.config.vocab_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.num_layers,
                            num_heads=self.config.num_heads,
                            max_context_length=self.config.max_context_length,
                            use_start_token=self.config.use_start_token,
                            key=key,
                        )
                    case "learned_additive":
                        return TransformerLearnedPE(
                            vocab_size=self.config.vocab_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.num_layers,
                            num_heads=self.config.num_heads,
                            max_context_length=self.config.max_context_length,
                            use_start_token=self.config.use_start_token,
                            key=key,
                        )
                    case "sine_additive":
                        return TransformerSinusoidalPE(
                            vocab_size=self.config.vocab_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.num_layers,
                            num_heads=self.config.num_heads,
                            max_context_length=self.config.max_context_length,
                            use_start_token=self.config.use_start_token,
                            key=key,
                        )
                    case _:
                        raise ValueError(f"Unknown position embedding type: {self.config.position_embedding_type}")
            case _:
                raise ValueError(f"Unknown model type: {self.config.model_type}")

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
        )

    def get_output(self, model: SequenceModel, batch: tuple[Array, Array], state: xax.State) -> Array:
        x_batched, _ = batch
        return jax.vmap(model.predict_sequence)(x_batched)

    def compute_loss(self, model: SequenceModel, batch: tuple[Array, Array], output: Array, state: xax.State) -> Array:
        (_, y), yhat = batch, output
        labels = jax.nn.one_hot(y, yhat.shape[-1])
        return optax.softmax_cross_entropy(logits=yhat, labels=labels).mean()

    def log_valid_step(
        self,
        model: SequenceModel,
        batch: tuple[Array, Array],
        output: Array,
        metrics: xax.FrozenDict[str, Array],
        state: xax.State,
    ) -> None:
        output_tokens = jnp.argmax(output, axis=-1)[0]
        output_words = "".join([self.ds.id_to_token[int(token)] for token in output_tokens])
        self.logger.log_string("teacher_forced_output", output_words)

        # Using the first few tokens from the batch, generate the rest of the sequence.
        prompt_seq = jnp.array([self.ds.token_to_id[c] for c in "To be"])
        generated_tokens = model.generate_sequence(prompt_seq, max_len=self.config.max_context_length)
        generated_words = "".join([self.ds.id_to_token[int(token)] for token in generated_tokens])
        self.logger.log_string("prompt", "".join([self.ds.id_to_token[int(token)] for token in prompt_seq]))
        self.logger.log_string("generated_output", generated_words)

    def get_data_iterator(self, phase: xax.Phase, key: PRNGKeyArray) -> Iterator[tuple[Array, Array]]:
        """Returns an iterator over batches of tokenized Shakespeare text.

        Args:
            phase: The phase of the data iterator to return.

        Returns:
            An iterator over batches of tokenized Shakespeare text, with
            each batch consisting of a tuple of the input tokens and the
            target tokens (shifted by one position).
        """
        seq_len = self.config.max_context_length
        # Split the token_ids into training and validation sets.
        if phase == "train":
            token_ids = self.token_ids[: int(0.95 * len(self.token_ids))]
        else:
            token_ids = self.token_ids[int(0.95 * len(self.token_ids)) :]
        n_tokens = token_ids.shape[0]

        while True:
            key, subkey = jax.random.split(key)
            # Sample starting indices for each sequence in the batch.
            idx = jax.random.randint(subkey, (self.config.batch_size,), 0, n_tokens - seq_len - 1)
            # Build the batch: for each starting index, extract a sequence of length seq_len + 1.
            batch_x = jnp.stack([token_ids[i : i + seq_len] for i in idx])
            batch_y = jnp.stack([token_ids[i + 1 : i + seq_len + 1] for i in idx])
            # One-hot encode the input sequences.
            yield batch_x, batch_y


if __name__ == "__main__":
    # Launch the training task.
    #   python -m train model_type=transformer position_embedding_type=rope use_start_token=False
    ShakespearePrediction.launch(Config())
