# distributed-training-lib

A from-scratch distributed training library built on PyTorch + NCCL.  
Goal: deeply understand Tensor Parallelism (TP) and Data Parallelism (DP) without relying on DTensor.

## Design principle

Don't wrap `torch.distributed` primitives — use them directly.  
Build abstractions for *topology*: which ranks belong to which TP/DP group, and which process group to pass to each collective.

```python
# Bad — unnecessary indirection
get_rank()          # just call dist.get_rank()

# Good — real abstraction
dims.tp_mesh        # DeviceMesh for the TP group *this rank* belongs to
dist.all_reduce(tensor, group=dims.tp_mesh.get_group())   # explicit PG
```

## Day 1 — Process Group Foundation

### File structure

```
distributed-training-lib/
├── core/
│   ├── __init__.py
│   ├── dist_init.py        # init/cleanup only, no wrappers
│   └── parallel_dims.py    # ParallelDims + DeviceMesh construction
├── tests/
│   ├── __init__.py
│   └── test_tp_communication.py   # all_reduce / all_gather / reduce_scatter on TP group
├── Makefile
└── README.md
```

### Quick start

```bash
cd distributed-training-lib

# Run all tests (uses 2 GPUs by default)
make test

# Explicit GPU count
make test NGPU=8

# Directly
torchrun --nproc_per_node=2 tests/test_tp_communication.py
```

### Expected output (2 GPUs)

```
[Rank 0] ParallelDims: tp=2, dp=1, world_size=2
[Rank 0] TP mesh: DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',))
[Rank 0] all_reduce on TP group PASSED: sum=1.0 (expected 1.0)
[Rank 0] all_gather on TP group PASSED: gathered=[0.0, 1.0]
[Rank 0] reduce_scatter on TP group PASSED (each rank holds its correct shard)
[Rank 0] All TP communication tests passed on 2 GPU(s)!
```

### API

```python
from core.dist_init import init_distributed, destroy_distributed
from core.parallel_dims import ParallelDims
import torch.distributed as dist

init_distributed()

dims = ParallelDims(tp=2, dp=1, world_size=dist.get_world_size())
dims.build_mesh()

# Use torch.distributed directly with the right process group
tensor = torch.randn(16, device=f"cuda:{dist.get_rank()}")
dist.all_reduce(tensor, group=dims.tp_mesh.get_group())   # TP collective

# Position within each parallelism group
dims.tp_rank     # this rank's index within its TP group
dims.tp_enabled  # True when tp > 1
```

### Key concepts

| Concept | What it means |
|---|---|
| `tp` | How many GPUs share the same model weights (sharded across them) |
| `dp` | How many independent model replicas run simultaneously |
| `tp_mesh` | The sub-mesh of GPUs in this rank's TP group — used for weight-shard collectives |
| `dp_mesh` | The sub-mesh of GPUs in this rank's DP group — used for gradient all-reduces |
| `reduce_scatter` | Core primitive for sequence-parallel & FSDP: reduce + shard in one step |

### Day 2 preview

Extend `ParallelDims` to a 2-D mesh (`[dp, tp]`) and implement a column-parallel linear layer whose weight is sharded along the TP dimension.
