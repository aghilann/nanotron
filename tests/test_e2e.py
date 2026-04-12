"""
Convergence sanity check — asserts loss drops significantly over training.

Single GPU:  python tests/test_e2e.py
2-GPU TP:    torchrun --nproc_per_node=2 tests/test_e2e.py
"""
import atexit
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from core.logger import DistLogger
from core.parallel_dims import ParallelDims
from models.gpt2.model import GPT2ForCausalLM
from models.gpt2.parallelize import build_model
from fixtures import TEXT, CONFIG, SEQ_LEN, LR, N_STEPS, assert_converged

BATCH_SIZE = 64


def _get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - SEQ_LEN, (BATCH_SIZE,), device=data.device)
    x = torch.stack([data[i     : i + SEQ_LEN    ] for i in ix])
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y


def _train(model: nn.Module, device: str, logger: DistLogger | None = None) -> tuple[float, float, float]:
    """Returns (initial_loss, final_loss, ms_per_step)."""
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
    assert_converged(initial, final)
    print("  PASSED", flush=True)


def test_tensor_parallel() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", local_rank),
    )
    atexit.register(dist.destroy_process_group)

    world_size = dist.get_world_size()
    dims = ParallelDims(tp=world_size, dp=1, world_size=world_size)
    dims.build_mesh()

    logger = DistLogger(only_log_rank=0)
    logger.log(f"=== tensor parallel ({world_size} GPUs) convergence ===")

    torch.manual_seed(42)
    model = build_model(CONFIG, dims, device="cuda")
    initial, final, ms = _train(model, device="cuda", logger=logger)

    logger.log(f"initial loss {initial:.4f} → final loss {final:.4f} | {ms:.1f} ms/step")
    assert_converged(initial, final)
    logger.log("PASSED")


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        test_tensor_parallel()
    else:
        test_single_gpu()
