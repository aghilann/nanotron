from __future__ import annotations

import torch
from torch import nn
from torch.distributed.nn.functional import all_reduce
from torch.distributed.device_mesh import DeviceMesh


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} not divisible by {denominator}"
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_mesh: DeviceMesh,
        bias: bool = False,
    ):
        super().__init__()
        self.tp_mesh = tp_mesh
        self.tp_group = tp_mesh.get_group()
        self.tp_rank = tp_mesh.get_local_rank()
        self.tp_size = tp_mesh.size()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

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
    """

    def __init__(self, input_size: int, output_size: int, tp_mesh: DeviceMesh, bias: bool = False):
        super().__init__(divide(input_size, tp_mesh.size()), output_size, tp_mesh, bias)

    def forward(self, x_BSH: torch.Tensor) -> torch.Tensor:
        y = self.linear(x_BSH)
        if self.tp_size > 1:
            y = all_reduce(y, group=self.tp_group)
        return y
