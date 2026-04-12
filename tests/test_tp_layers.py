"""
torchrun --nproc_per_node=2 tests/test_tp_layers.py
"""
import atexit
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from core.logger import DistLogger
from core.parallel_dims import ParallelDims
from core.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, ParallelLMHead

DEVICE = torch.device("cuda")


def shard_col(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Slice output dim (dim 0) for this rank."""
    chunk = weight.shape[0] // tp_size
    return weight[tp_rank * chunk : (tp_rank + 1) * chunk]


def shard_row(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Slice input dim (dim 1) for this rank."""
    chunk = weight.shape[1] // tp_size
    return weight[:, tp_rank * chunk : (tp_rank + 1) * chunk]


def test_column_parallel(dims: ParallelDims, logger: DistLogger) -> None:
    torch.manual_seed(0)
    ref = nn.Linear(8, 8, bias=False)
    x = torch.randn(2, 8, device=DEVICE)

    col = ColumnParallelLinear(8, 8, dims.tp_mesh, bias=False).to(DEVICE)
    col.linear.weight.data.copy_(shard_col(ref.weight.data, dims.tp_rank, dims.tp).to(DEVICE))

    # Output is sharded — gather across ranks to reconstruct full output
    local_out = col(x)
    gathered = [torch.zeros_like(local_out) for _ in range(dims.tp)]
    dist.all_gather(gathered, local_out, group=dims.tp_mesh.get_group())
    full_out = torch.cat(gathered, dim=-1)

    expected = ref(x.cpu()).to(DEVICE)
    assert torch.allclose(full_out, expected, atol=1e-5), \
        f"ColumnParallel FAILED: max diff {(full_out - expected).abs().max():.2e}"
    logger.log("ColumnParallelLinear PASSED")


def test_row_parallel(dims: ParallelDims, logger: DistLogger) -> None:
    torch.manual_seed(0)
    ref = nn.Linear(8, 8, bias=False)
    x = torch.randn(2, 8)

    row = RowParallelLinear(8, 8, dims.tp_mesh, bias=False).to(DEVICE)
    row.linear.weight.data.copy_(shard_row(ref.weight.data, dims.tp_rank, dims.tp).to(DEVICE))

    chunk = 8 // dims.tp
    x_shard = x[:, dims.tp_rank * chunk : (dims.tp_rank + 1) * chunk].to(DEVICE)

    out = row(x_shard)  # all_reduce inside → full output
    expected = ref(x).to(DEVICE)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"RowParallel FAILED: max diff {(out - expected).abs().max():.2e}"
    logger.log("RowParallelLinear PASSED")


def test_sharded_block(dims: ParallelDims, logger: DistLogger) -> None:
    """Column → Row output matches two sequential reference nn.Linears."""
    torch.manual_seed(0)
    ref1 = nn.Linear(8, 8, bias=False)
    ref2 = nn.Linear(8, 8, bias=False)
    x = torch.randn(2, 8, device=DEVICE)

    col = ColumnParallelLinear(8, 8, dims.tp_mesh, bias=False).to(DEVICE)
    row = RowParallelLinear(8, 8, dims.tp_mesh, bias=False).to(DEVICE)
    col.linear.weight.data.copy_(shard_col(ref1.weight.data, dims.tp_rank, dims.tp).to(DEVICE))
    row.linear.weight.data.copy_(shard_row(ref2.weight.data, dims.tp_rank, dims.tp).to(DEVICE))

    out = row(col(x))
    expected = ref2(ref1(x.cpu())).to(DEVICE)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"ShardedBlock FAILED: max diff {(out - expected).abs().max():.2e}"
    logger.log("ShardedBlock (Column → Row) PASSED")


def shard_vocab(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Slice vocab dim (dim 0) for this rank."""
    chunk = weight.shape[0] // tp_size
    return weight[tp_rank * chunk : (tp_rank + 1) * chunk]


def test_vocab_parallel_embedding(dims: ParallelDims, logger: DistLogger) -> None:
    torch.manual_seed(0)
    VOCAB, DIM = 32, 8
    ref = nn.Embedding(VOCAB, DIM)
    x = torch.tensor([0, 1, 7, 8, 16, 31], device=DEVICE)

    emb = VocabParallelEmbedding(VOCAB, DIM, dims.tp_mesh).to(DEVICE)
    emb.weight.data.copy_(shard_vocab(ref.weight.data, dims.tp_rank, dims.tp).to(DEVICE))

    out = emb(x)
    expected = ref(x.cpu()).to(DEVICE)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"VocabParallelEmbedding FAILED: max diff {(out - expected).abs().max():.2e}"
    logger.log("VocabParallelEmbedding PASSED")


def test_parallel_lm_head(dims: ParallelDims, logger: DistLogger) -> None:
    torch.manual_seed(0)
    VOCAB, DIM = 32, 8
    ref = nn.Linear(DIM, VOCAB, bias=False)
    x = torch.randn(2, DIM, device=DEVICE)

    head = ParallelLMHead(VOCAB, DIM, dims.tp_mesh).to(DEVICE)
    head.weight.data.copy_(shard_vocab(ref.weight.data, dims.tp_rank, dims.tp).to(DEVICE))

    out = head(x)
    expected = ref(x.cpu()).to(DEVICE)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"ParallelLMHead FAILED: max diff {(out - expected).abs().max():.2e}"
    logger.log("ParallelLMHead PASSED")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
    dist.init_process_group(timeout=timedelta(seconds=120), device_id=torch.device("cuda", local_rank))
    atexit.register(dist.destroy_process_group)

    world_size = dist.get_world_size()
    dims = ParallelDims(tp=world_size, dp=1, world_size=world_size)
    dims.build_mesh()

    logger = DistLogger()
    logger.log(str(dims))

    test_column_parallel(dims, logger)
    test_row_parallel(dims, logger)
    test_sharded_block(dims, logger)
    test_vocab_parallel_embedding(dims, logger)
    test_parallel_lm_head(dims, logger)

    logger.log("All layer tests passed!")


if __name__ == "__main__":
    main()
