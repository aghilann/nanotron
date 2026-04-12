"""
DP+TP benchmark: compare different TP/DP splits on 8 GPUs.

Both configs process the same GLOBAL_BATCH * SEQ_LEN tokens per step —
only how that work is divided changes.

  TP=4, DP=2:  4-way model sharding, 2 data replicas, local_batch=64
  TP=2, DP=4:  2-way model sharding, 4 data replicas, local_batch=32

torchrun --nproc_per_node=8 tests/test_dp_tp_benchmark.py --tp 4 --dp 2
torchrun --nproc_per_node=8 tests/test_dp_tp_benchmark.py --tp 2 --dp 4
"""
import argparse
import atexit
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.logger import DistLogger
from core.parallel_dims import ParallelDims
from models.gpt2.parallelize import build_model
from fixtures import TEXT, CONFIG, SEQ_LEN, LR, N_STEPS, assert_converged

GLOBAL_BATCH = 128   # total samples/step summed across all DP ranks


def _get_batch(
    data: torch.Tensor,
    local_batch: int,
    step: int,
    dp_rank: int,
    dp: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    All ranks use the same seed so they agree on the global index pool.
    Each DP rank then takes its own contiguous slice — different data,
    same total tokens as every other configuration.
    """
    torch.manual_seed(step)
    ix = torch.randint(0, len(data) - SEQ_LEN, (GLOBAL_BATCH,), device=data.device)
    ix = ix[dp_rank * local_batch : (dp_rank + 1) * local_batch]
    x = torch.stack([data[i     : i + SEQ_LEN    ] for i in ix])
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y


def train(model, dims: ParallelDims, logger: DistLogger) -> tuple[float, float, float]:
    """Returns (initial_loss, final_loss, ms_per_step)."""
    data        = torch.tensor(list(TEXT.encode("utf-8")), dtype=torch.long, device="cuda")
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
    local_batch = GLOBAL_BATCH // dims.dp

    initial_loss = final_loss = None
    start = time.perf_counter()

    for step in range(1, N_STEPS + 1):
        x, y = _get_batch(data, local_batch, step, dims.dp_rank, dims.dp)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, CONFIG.vocab_size), y.view(-1))
        loss.backward()   # DDP all-reduces grads across DP; TP all-reduce already in fwd
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()
        final_loss = loss.item()

        if step % 25 == 0:
            logger.log(f"step {step:4d} | loss {final_loss:.4f}")

    ms_per_step = (time.perf_counter() - start) / N_STEPS * 1000
    return initial_loss, final_loss, ms_per_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dp", type=int, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
    dist.init_process_group(
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", local_rank),
    )
    atexit.register(dist.destroy_process_group)

    world_size = dist.get_world_size()
    dims = ParallelDims(tp=args.tp, dp=args.dp, world_size=world_size)
    dims.build_mesh()

    logger = DistLogger(only_log_rank=0)
    tokens_per_step = GLOBAL_BATCH * SEQ_LEN
    logger.log(f"=== TP={args.tp} DP={args.dp} | {world_size} GPUs | {tokens_per_step} tokens/step ===")
    logger.log(f"  local_batch={GLOBAL_BATCH // args.dp} per DP rank")

    torch.manual_seed(42)
    model = build_model(CONFIG, dims, device="cuda")

    initial, final, ms = train(model, dims, logger)

    tokens_per_sec = int(tokens_per_step / (ms / 1000))
    logger.log(f"initial loss {initial:.4f} → final loss {final:.4f}")
    logger.log(f"{ms:.1f} ms/step | {tokens_per_sec:,} tokens/sec")
    assert_converged(initial, final)
    logger.log("PASSED")


if __name__ == "__main__":
    main()
