import torch
import torch.distributed as dist

from core.logger import DistLogger
from core.parallel_dims import ParallelDims
from utils import init_dist

DEVICE = torch.device("cuda")


# Sum of 0+1+...+(tp-1). With tp=2: 1, with tp=8: 28.
def tp_sum(tp: int) -> int:
    return tp * (tp - 1) // 2


def test_tp_all_reduce(dims: ParallelDims, logger: DistLogger) -> None:
    # Each rank puts its tp_rank into a tensor; all_reduce SUM adds them so every rank ends up with the total.
    # tp=2: rank0=[0], rank1=[1] → both get [1]
    tensor = torch.tensor([dims.tp_rank], dtype=torch.int32, device=DEVICE)
    expected = torch.tensor([tp_sum(dims.tp)], dtype=torch.int32, device=DEVICE)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dims.tp_mesh.get_group())

    assert torch.allclose(tensor.float(), expected.float()), \
        f"all_reduce: got {tensor.item()}, expected {expected.item()}"
    logger.log(f"all_reduce PASSED: sum={tensor.item()} (expected {expected.item()})")


def test_tp_all_gather(dims: ParallelDims, logger: DistLogger) -> None:
    # Each rank contributes its tp_rank; all_gather assembles the full list on every rank.
    # tp=2: rank0=[0], rank1=[1] → both get [0, 1]
    local = torch.tensor([dims.tp_rank], dtype=torch.int32, device=DEVICE)
    gathered = [torch.empty(1, dtype=torch.int32, device=DEVICE) for _ in range(dims.tp)]

    dist.all_gather(gathered, local, group=dims.tp_mesh.get_group())

    result = torch.cat(gathered)
    expected = torch.arange(dims.tp, dtype=torch.int32, device=DEVICE)
    assert torch.allclose(result.float(), expected.float()), \
        f"all_gather: got {result.tolist()}, expected {expected.tolist()}"
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
    assert torch.allclose(output.float(), expected.float()), \
        f"reduce_scatter: tp_rank={dims.tp_rank} got {output.item()}, expected {expected.item()}"
    dist.barrier(group=dims.tp_mesh.get_group())
    logger.log("reduce_scatter PASSED (each rank holds its correct shard)")


def main() -> None:
    dims, logger = init_dist()
    logger.log(str(dims))
    logger.log(f"TP mesh: {dims.tp_mesh}")

    test_tp_all_reduce(dims, logger)
    test_tp_all_gather(dims, logger)
    test_tp_reduce_scatter(dims, logger)

    logger.log(f"All TP communication tests passed on {dims.tp} GPU(s)!")


if __name__ == "__main__":
    main()
