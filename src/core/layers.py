import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather, all_reduce
from torch.distributed.device_mesh import DeviceMesh


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} not divisible by {denominator}"
    return numerator // denominator


class TPModule(nn.Module):
    """Base class that extracts and caches TP mesh metadata."""

    def __init__(self, tp_mesh: DeviceMesh) -> None:
        super().__init__()
        self.tp_mesh = tp_mesh
        self.tp_group = tp_mesh.get_group()
        self.tp_rank = tp_mesh.get_local_rank()
        self.tp_size = tp_mesh.size()


class LinearBase(TPModule):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_mesh: DeviceMesh,
        bias: bool = False,
    ):
        super().__init__(tp_mesh)
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):
    """
    Linear sharded along the output dimension.
    Weight: [output_size/tp, input_size] per rank.
    No comms in forward — caller (e.g. ShardedMLP) is responsible.
    """

    def __init__(self, input_size: int, output_size: int, tp_mesh: DeviceMesh, bias: bool = False):
        super().__init__(input_size, divide(output_size, tp_mesh.size()), tp_mesh, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(LinearBase):
    """
    Linear sharded along the input dimension.
    Weight: [output_size, input_size/tp] per rank.
    All-reduce in forward combines partial sums across TP ranks.
    Bias is replicated and added once after the all-reduce, not inside the
    sharded linear (otherwise it would be accumulated tp_size times).
    """

    def __init__(self, input_size: int, output_size: int, tp_mesh: DeviceMesh, bias: bool = False):
        super().__init__(divide(input_size, tp_mesh.size()), output_size, tp_mesh, bias=False)
        self._bias = nn.Parameter(torch.zeros(output_size)) if bias else None

    @property
    def bias(self) -> nn.Parameter | None:
        return self._bias

    def forward(self, x_BSH: torch.Tensor) -> torch.Tensor:
        y = self.linear(x_BSH)
        if self.tp_size > 1:
            y = all_reduce(y, group=self.tp_group)
        if self._bias is not None:
            y = y + self._bias
        return y


class VocabParallelEmbedding(TPModule):
    """
    Embedding table sharded along the vocabulary dimension.
    Each rank holds [num_embeddings/tp, embedding_dim] rows.
    Tokens outside a rank's range are masked to 0 before lookup;
    all_reduce sums the partial results so every rank gets the full embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_mesh: DeviceMesh,
    ):
        super().__init__(tp_mesh)
        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x_local = mask * (x - self.vocab_start_idx)
        else:
            x_local = x
        y = F.embedding(x_local, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            y = all_reduce(y, group=self.tp_group)
        return y


class ParallelLMHead(TPModule):
    """
    LM head sharded along the vocabulary (output) dimension.
    Weight: [vocab/tp, embedding_dim] per rank.
    Each rank computes partial logits; all_gather reconstructs
    the full [*, vocab] tensor on every rank.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_mesh: DeviceMesh,
    ):
        super().__init__(tp_mesh)
        self.weight = nn.Parameter(torch.empty(divide(num_embeddings, self.tp_size), embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(x, self.weight)  # [..., vocab/tp]
        if self.tp_size == 1:
            return local_logits
        return torch.cat(all_gather(local_logits, group=self.tp_group), dim=-1)
