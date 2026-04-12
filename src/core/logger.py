import torch.distributed as dist


class DistLogger:
    def __init__(self, only_log_rank: int | None = None):
        self.only_log_rank = only_log_rank

    def log(self, msg: str) -> None:
        rank = dist.get_rank()
        if self.only_log_rank is None or rank == self.only_log_rank:
            print(f"[GPU {rank}] {msg}", flush=True)
