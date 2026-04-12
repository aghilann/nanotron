import atexit
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from core.logger import DistLogger
from core.parallel_dims import ParallelDims


def init_dist() -> tuple[ParallelDims, DistLogger]:
    """Initialize the distributed process group for a pure-TP run."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
    dist.init_process_group(
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", local_rank),
    )
    atexit.register(dist.destroy_process_group)

    world_size = dist.get_world_size()
    dims = ParallelDims(tp=world_size, dp=1, world_size=world_size)
    dims.build_mesh()

    return dims, DistLogger()


def shard(weight: torch.Tensor, rank: int, size: int, dim: int = 0) -> torch.Tensor:
    """Slice a weight tensor along `dim` for this rank."""
    chunk = weight.shape[dim] // size
    idx = [slice(None)] * weight.dim()
    idx[dim] = slice(rank * chunk, (rank + 1) * chunk)
    return weight[tuple(idx)]
