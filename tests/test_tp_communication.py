import atexit
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from core.logger import DistLogger
from core.parallel_dims import ParallelDims

DEVICE = torch.device("cuda")  # set_device() in main() makes this per-process


# Sum of 0+1+...+(tp-1). With tp=2: 1, with tp=8: 28.
def tp_sum(tp: int) -> int:
    return tp * (tp - 1) // 2


def abort_all(msg: str) -> None:
    print(f"[GPU {dist.get_rank()}] FAILED: {msg}", flush=True, file=sys.stderr)
    sys.exit(1)


def test_tp_all_reduce(dims: ParallelDims, logger: DistLogger) -> None:
    # Each rank puts its tp_rank into a tensor; all_reduce SUM adds them so every rank ends up with the total.
    # tp=2: rank0=[0], rank1=[1] → both get [1]
    tensor = torch.tensor([dims.tp_rank], dtype=torch.int32, device=DEVICE)
    expected = torch.tensor([tp_sum(dims.tp)], dtype=torch.int32, device=DEVICE)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dims.tp_mesh.get_group())

    if not torch.allclose(tensor.float(), expected.float()):
        abort_all(f"all_reduce: got {tensor.item()}, expected {expected.item()}")
    logger.log(f"all_reduce PASSED: sum={tensor.item()} (expected {expected.item()})")


def test_tp_all_gather(dims: ParallelDims, logger: DistLogger) -> None:
    # Each rank contributes its tp_rank; all_gather assembles the full list on every rank.
    # tp=2: rank0=[0], rank1=[1] → both get [0, 1]
    local = torch.tensor([dims.tp_rank], dtype=torch.int32, device=DEVICE)
    gathered = [torch.empty(1, dtype=torch.int32, device=DEVICE) for _ in range(dims.tp)]

    dist.all_gather(gathered, local, group=dims.tp_mesh.get_group())

    result = torch.cat(gathered)
    expected = torch.arange(dims.tp, dtype=torch.int32, device=DEVICE)
    if not torch.allclose(result.float(), expected.float()):
        abort_all(f"all_gather: got {result.tolist()}, expected {expected.tolist()}")
    logger.log(f"all_gather PASSED: gathered={result.tolist()}")


def test_tp_reduce_scatter(dims: ParallelDims, logger: DistLogger) -> None:
    # reduce_scatter = all_reduce + split: each rank gets only its own shard of the result.
    # Used in FSDP and sequence-parallel. Input element i = tp_rank + i on every rank.
    # tp=2: rank0 sends [0,1], rank1 sends [1,2] → reduced [1,3] → rank0 gets [1], rank1 gets [3]
    tp = dims.tp
    input_tensor = torch.tensor(
        [dims.tp_rank + i for i in range(tp)], dtype=torch.int32, device=DEVICE
    )
    output = torch.empty(1, dtype=torch.int32, device=DEVICE)

    dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM, group=dims.tp_mesh.get_group())

    # Rank r owns element r of the reduced vector: tp_sum + r*tp
    expected = torch.tensor([tp_sum(tp) + dims.tp_rank * tp], dtype=torch.int32, device=DEVICE)
    if not torch.allclose(output.float(), expected.float()):
        abort_all(f"reduce_scatter: tp_rank={dims.tp_rank} got {output.item()}, expected {expected.item()}")
    dist.barrier(group=dims.tp_mesh.get_group())
    logger.log(f"reduce_scatter PASSED (each rank holds its correct shard)")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    # Bind this process to its GPU before init_process_group so NCCL uses the right device.
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
    dist.init_process_group(timeout=timedelta(seconds=120), device_id=torch.device("cuda", local_rank))
    atexit.register(dist.destroy_process_group)

    # All GPUs form one TP group, no data parallelism yet.
    world_size = dist.get_world_size()
    dims = ParallelDims(tp=world_size, dp=1, world_size=world_size)
    dims.build_mesh()

    logger = DistLogger()
    logger.log(str(dims))
    logger.log(f"TP mesh: {dims.tp_mesh}")

    test_tp_all_reduce(dims, logger)
    test_tp_all_gather(dims, logger)
    test_tp_reduce_scatter(dims, logger)

    logger.log(f"All TP communication tests passed on {world_size} GPU(s)!")


if __name__ == "__main__":
    main()
