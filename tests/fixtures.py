"""Shared constants used across training tests."""
from models.gpt2.model import GPT2Config

TEXT = """\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them.\
""" * 2

CONFIG = GPT2Config(
    vocab_size=256,
    hidden_size=8192,
    num_layers=4,
    num_heads=32,
    max_seq_len=128,
    mlp_ratio=4,
)

SEQ_LEN    = 128
LR         = 3e-3
N_STEPS    = 200
LOSS_RATIO = 0.5   # final loss must be at most this fraction of initial loss
LOSS_CEIL  = 4.0   # final loss must be below this absolute value


def assert_converged(initial: float, final: float) -> None:
    assert final < initial * LOSS_RATIO, f"Loss did not halve: {initial:.4f} → {final:.4f}"
    assert final < LOSS_CEIL,            f"Final loss too high: {final:.4f} >= {LOSS_CEIL}"
