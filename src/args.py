from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskArguments:
    """
    Arguments for running the task.
    """
    glove_emb: bool = field(
        default=False,
    )
    tasks: list[str] = field(
        default=None,
        metadata={"help": "tasks to run"},
    )
    TODO: str = field(
        default="train+analyze",
    )


@dataclass
class CTRNNModelArguments:
    """
    Arguments for running the continuous-time RNN model.
    """
    CTRNN: Optional[bool] = field(
        default=True,
        metadata={"help": "If flag is set, running with continuous-time CTRNN, else running with discrete-time RNN"},
    )
    nonlinearity: Optional[str] = field(
        default='relu',
        metadata={"help": "activation function used in CTRNN model, can be one of relu, softplus, ..."},
    )
    sparse_model: Optional[bool] = field(
        default=False,
        metadata={"help": "If flag is set, we apply a mask to the h2h layer of the CTRNN a la Khona, Chandra et al. 2022"},
    )
    sigma_rec: Optional[float] = field(
        default=0.05,
        metadata={"help": "recurrent unit noise"},
    )


@dataclass
class RNNModelArguments:
    """
    Arguments for running the discrete-time RNN model.
    """
    discrete_time_rnn: Optional[str] = field(
        default=None,
        metadata={"help": "type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, RNN_SOFTPLUS)"},
    )
    nlayers: Optional[int] = field(
        default=1,
        metadata={"help": "number of layers"},
    )
    dropout: Optional[float] = field(
        default=0.2,
        metadata={"help": "dropout applied to layers (0 = no dropout)"},
    )
    tie_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "tie the word embedding and softmax weights"},
    )


@dataclass
class SharedModelArguments:
    """
    Arguments for running the either model.
    """
    hidden_size: int = field(
        default=256,  # FIXME 256 = 16*16 #Change to 300 if GloVe
        metadata={"help": "number of hidden units per layer"},
    )


@dataclass
class TrainingArguments:
    """
    Arguments for running the training.
    """
    dt: int = field(
        default=100,
        metadata={"help": "neuronal time constant"},
    )
    epochs: int = field(
        default=5,
        metadata={"help": "number of epochs"},
    )
    batch_size: int = field(
        default=20,
        metadata={"help": "batch size"},
    )
    eval_batch_size: int = field(
        default=10,
        metadata={"help": "eval batch size"},
    )
    bptt: int = field(
        default=35,
        metadata={"help": "sequence length"},
    )
    seq_len: int = field(
        default=100,
        metadata={"help": "sequence length for cog tasks"},
    )
    lr: float = field(
        default=1e-3,
        metadata={"help": "initial learning rate"},
    )
    clip: float = field(
        default=0.25,
        metadata={"help": "gradient clipping"},
    )
    seed: int = field(
        default=1,
    )
    cuda: bool = field(
        default=False,
    )
    save: str = field(
        default='model.pt',
    )
    dry_run: bool = field(
        default=False,
    )
    log_level: str = field(
        default='INFO',
    )
    log_interval: int = field(
        default=200,
    )
    dry_run: bool = field(
        default=False,
    )
    training_yang: int = field(
        default=8000,
    )

