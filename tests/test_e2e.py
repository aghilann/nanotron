"""
Convergence sanity check — asserts loss drops significantly over training.

Single GPU:  python tests/test_e2e.py
2-GPU TP:    torchrun --nproc_per_node=2 tests/test_e2e.py
"""
import atexit
import os
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch import nn

from core.logger import DistLogger
from models.gpt2.model import GPT2Config, GPT2ForCausalLM
from models.gpt2.parallelize import build_model

# Tiny Shakespeare excerpt — ~800 bytes, enough to overfit
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

BATCH_SIZE  = 64
SEQ_LEN     = 128
LR          = 3e-3
N_STEPS     = 200
LOSS_RATIO  = 0.5   # final loss must be at most 50% of initial loss
LOSS_CEIL   = 4.0   # final loss must be below this absolute value


def _get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - SEQ_LEN, (BATCH_SIZE,), device=data.device)
    x = torch.stack([data[i     : i + SEQ_LEN    ] for i in ix])
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y


def _train(model: nn.Module, device: str, logger: DistLogger | None = None) -> tuple[float, float, float]:
    """Returns (initial_loss, final_loss, ms_per_step)."""
    import time
    data = torch.tensor(list(TEXT.encode("utf-8")), dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    initial_loss = None
    loss = None
    start = time.perf_counter()

    for step in range(1, N_STEPS + 1):
        x, y = _get_batch(data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, CONFIG.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()
        if logger is not None and step % 25 == 0:
            logger.log(f"step {step:4d} | loss {loss.item():.4f}")

    ms_per_step = (time.perf_counter() - start) / N_STEPS * 1000
    return initial_loss, loss.item(), ms_per_step


def test_single_gpu() -> None:
    print("=== single GPU convergence ===", flush=True)
    torch.manual_seed(42)
    model = GPT2ForCausalLM(CONFIG).to("cuda")
    initial, final, ms = _train(model, device="cuda")
    print(f"  initial loss {initial:.4f} → final loss {final:.4f} | {ms:.1f} ms/step", flush=True)
    assert final < initial * LOSS_RATIO, f"Loss did not halve: {initial:.4f} → {final:.4f}"
    assert final < LOSS_CEIL,            f"Final loss too high: {final:.4f} >= {LOSS_CEIL}"
    print("  PASSED", flush=True)


def test_tensor_parallel() -> None:
    import torch.distributed as dist
    from core.parallel_dims import ParallelDims

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", local_rank),
    )
    atexit.register(dist.destroy_process_group)

    world_size = dist.get_world_size()
    logger = DistLogger(only_log_rank=0)
    logger.log(f"=== tensor parallel ({world_size} GPUs) convergence ===")

    dims = ParallelDims(tp=world_size, dp=1, world_size=world_size)
    dims.build_mesh()

    torch.manual_seed(42)
    model = build_model(CONFIG, dims.tp_mesh, device="cuda")
    initial, final, ms = _train(model, device="cuda", logger=logger)

    logger.log(f"initial loss {initial:.4f} → final loss {final:.4f} | {ms:.1f} ms/step")
    assert final < initial * LOSS_RATIO, f"Loss did not halve: {initial:.4f} → {final:.4f}"
    assert final < LOSS_CEIL,            f"Final loss too high: {final:.4f} >= {LOSS_CEIL}"
    logger.log("PASSED")


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        test_tensor_parallel()
    else:
        test_single_gpu()
