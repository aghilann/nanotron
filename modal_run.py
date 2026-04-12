# modal_run.py
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "nanotron-runner"
CUDA_BASE = "nvidia/cuda:12.6.3-devel-ubuntu22.04"

REMOTE_PROJECT_DIR = "/workspace/project"
GPU = "H100"
TIMEOUT_S = 60 * 60
DEFAULT_CMD = "make test-tp"
DEFAULT_NGPU = 2

EXCLUDED_NAMES = {
    ".venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".DS_Store",
    "node_modules",
}

app = modal.App(APP_NAME)


def _find_project_root() -> Path:
    candidates = (
        Path(__file__).resolve().parent,
        Path(REMOTE_PROJECT_DIR),
        Path.cwd(),
    )
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Expected pyproject.toml in one of: "
        + ", ".join(str(candidate / "pyproject.toml") for candidate in candidates)
    )


PROJECT_ROOT = _find_project_root()


def _ignore_path(path: Path) -> bool:
    p = Path(path)
    if any(part in EXCLUDED_NAMES for part in p.parts):
        return True
    return p.name.endswith(".egg-info")


def _must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}")
    return p


PYPROJECT = _must_exist(PROJECT_ROOT / "pyproject.toml")
UV_LOCK = _must_exist(PROJECT_ROOT / "uv.lock")
README = PROJECT_ROOT / "README.md"  # declared in pyproject.toml

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends bash make git && rm -rf /var/lib/apt/lists/*",
        "python -m pip install -U pip uv",
    )
    .add_local_file(str(PYPROJECT), f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .add_local_file(str(UV_LOCK), f"{REMOTE_PROJECT_DIR}/uv.lock", copy=True)
    .add_local_file(str(README), f"{REMOTE_PROJECT_DIR}/README.md", copy=True)
    .run_commands(
        f"cd {REMOTE_PROJECT_DIR} && uv sync --frozen",
    )
    # Ship full source at container start — fast iteration without image rebuilds
    .add_local_dir(
        local_path=str(PROJECT_ROOT),
        remote_path=REMOTE_PROJECT_DIR,
        ignore=_ignore_path,
        copy=False,
    )
)


def _exec(cmd: str, n_gpus: int) -> None:
    venv_bin = f"{REMOTE_PROJECT_DIR}/.venv/bin"
    env = os.environ.copy()
    # NGPU always matches the Modal hardware allocation so torchrun never
    # tries to open a device ordinal that doesn't exist.
    env["NGPU"] = str(n_gpus)
    env["PYTHONPATH"] = f"{REMOTE_PROJECT_DIR}/src:{env.get('PYTHONPATH', '')}"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = f"{REMOTE_PROJECT_DIR}/.venv"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(
        ["bash", "-lc", f"cd {REMOTE_PROJECT_DIR} && {cmd}"],
        check=True,
        env=env,
    )


# One Modal function per GPU count — GPU config must be fixed at decoration time.
@app.function(image=image, gpu=f"{GPU}:1", timeout=TIMEOUT_S)
def run_1gpu(cmd: str) -> None:
    _exec(cmd, 1)


@app.function(image=image, gpu=f"{GPU}:2", timeout=TIMEOUT_S)
def run_2gpu(cmd: str) -> None:
    _exec(cmd, 2)


@app.function(image=image, gpu=f"{GPU}:4", timeout=TIMEOUT_S)
def run_4gpu(cmd: str) -> None:
    _exec(cmd, 4)


@app.function(image=image, gpu=f"{GPU}:8", timeout=TIMEOUT_S)
def run_8gpu(cmd: str) -> None:
    _exec(cmd, 8)


# Convenience alias for ad-hoc runs (defaults to 2 GPUs).
@app.function(image=image, gpu=f"{GPU}:{DEFAULT_NGPU}", timeout=TIMEOUT_S)
def run(cmd: str) -> None:
    _exec(cmd, DEFAULT_NGPU)


@app.local_entrypoint()
def main(
    cmd: str = DEFAULT_CMD,
    ngpu: int = DEFAULT_NGPU,
    benchmark: bool = False,
    benchmark_dp: bool = False,
) -> None:
    """
    Run a command on Modal GPUs.

    Examples:
        modal run modal_run.py
        modal run modal_run.py --cmd "make test-layers"
        modal run modal_run.py --ngpu 8 --cmd "torchrun --nproc_per_node=8 tests/test_e2e.py"
        modal run modal_run.py --benchmark           # 4-GPU vs 8-GPU TP sweep in parallel
        modal run modal_run.py --benchmark-dp        # TP=4,DP=2 vs TP=2,DP=4 in parallel
    """
    _fn = {1: run_1gpu, 2: run_2gpu, 4: run_4gpu, 8: run_8gpu}.get(ngpu, run)
    with modal.enable_output():
        if benchmark_dp:
            handles = [
                run_8gpu.spawn(cmd="torchrun --nproc_per_node=8 tests/test_dp_tp_benchmark.py --tp 4 --dp 2"),
                run_8gpu.spawn(cmd="torchrun --nproc_per_node=8 tests/test_dp_tp_benchmark.py --tp 2 --dp 4"),
            ]
            for handle in handles:
                handle.get()
        elif benchmark:
            handles = [
                run_4gpu.spawn(cmd="torchrun --nproc_per_node=4 tests/test_e2e.py"),
                run_8gpu.spawn(cmd="torchrun --nproc_per_node=8 tests/test_e2e.py"),
            ]
            for handle in handles:
                handle.get()
        else:
            _fn.remote(cmd=cmd)
