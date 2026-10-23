"""Local source/runtime identity for diagnostic runs, with no network operations."""
from importlib.metadata import version, distribution, PackageNotFoundError
import hashlib
import os
from pathlib import Path
import platform
import subprocess


def source_runtime(root: Path, execution_device="cpu"):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(set(p for p in paths if p)):
        p = root / raw.decode()
        if p.is_file():
            digest.update(raw + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    packages = {}
    for name in ("numpy", "torch", "mujoco", "warp-lang", "gymnasium", "skrl"):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    origins = {}
    for name in ("comfree_warp", "newton"):
        try:
            packages[name] = version(name)
            origins[name] = distribution(name).read_text("direct_url.json")
        except PackageNotFoundError:
            pass
    device = {'requested': str(execution_device)}
    if str(execution_device).startswith('cuda'):
        import torch
        props = torch.cuda.get_device_properties(execution_device)
        device.update(name=props.name, total_memory_bytes=props.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(execution_device)),
                      torch_cuda_runtime=torch.version.cuda,
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return {"source_commit": git("rev-parse", "HEAD").decode().strip(),
            "source_tree_sha256": digest.hexdigest(),
            "git_status": git("status", "--short").decode(),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "package_origins": origins,
            "execution_device": str(execution_device),
            "device_identity": device,
            "scheduler": {key: os.environ[key] for key in
                          ('SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'PBS_JOBID', 'PBS_ARRAY_INDEX') if key in os.environ},
            "gpu_validation": "requested_bounded_interface_check" if str(execution_device).startswith("cuda") else "not_performed"}
