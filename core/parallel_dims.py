from __future__ import annotations

from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


@dataclass
class ParallelDims:
    tp: int
    dp: int
    world_size: int

    _world_mesh: DeviceMesh | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.tp >= 1, "tp must be >= 1"
        assert self.dp >= 1, "dp must be >= 1"
        assert self.tp * self.dp == self.world_size, (
            f"tp({self.tp}) * dp({self.dp}) = {self.tp * self.dp} "
            f"!= world_size({self.world_size})"
        )

    def build_mesh(self) -> DeviceMesh:
        if self._world_mesh is not None:
            return self._world_mesh

        if self.dp == 1:
            self._world_mesh = init_device_mesh("cuda", (self.tp,), mesh_dim_names=("tp",))
        else:
            # Outer axis DP, inner axis TP
            self._world_mesh = init_device_mesh(
                "cuda", (self.dp, self.tp), mesh_dim_names=("dp", "tp")
            )

        return self._world_mesh

    @property
    def tp_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            raise RuntimeError("Call build_mesh() before accessing tp_mesh.")
        return self._world_mesh if self.dp == 1 else self._world_mesh["tp"]

    @property
    def dp_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            raise RuntimeError("Call build_mesh() before accessing dp_mesh.")
        if not self.dp_enabled:
            raise ValueError("dp_mesh is not available when dp=1.")
        return self._world_mesh["dp"]

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp > 1

    @property
    def tp_rank(self) -> int:
        if self._world_mesh is None:
            raise RuntimeError("Call build_mesh() before querying tp_rank.")
        return self.tp_mesh.get_local_rank()

    @property
    def dp_rank(self) -> int:
        if self._world_mesh is None:
            raise RuntimeError("Call build_mesh() before querying dp_rank.")
        return self.dp_mesh.get_local_rank() if self.dp_enabled else 0

    def __str__(self) -> str:
        return f"ParallelDims: tp={self.tp}, dp={self.dp}, world_size={self.world_size}"
