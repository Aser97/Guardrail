from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def resolve_device_from_hackathon(project_root: Path) -> str:
    """
    Resolve classifier device from hackathon.json.

    Policy:
    - needs_gpu=true  -> require CUDA, fail fast if unavailable
    - needs_gpu=false -> force CPU
    - missing/invalid file -> fail fast
    """
    hackathon_path = (project_root.parent / "hackathon.json").resolve()
    if not hackathon_path.exists():
        raise RuntimeError(f"Required config not found: {hackathon_path}")

    try:
        cfg = json.loads(hackathon_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON in {hackathon_path}: {exc}")

    if not isinstance(cfg, dict):
        raise RuntimeError(f"{hackathon_path} must contain a JSON object")
    if "needs_gpu" not in cfg:
        raise RuntimeError(f"{hackathon_path} missing required field: needs_gpu")
    if not isinstance(cfg["needs_gpu"], bool):
        raise RuntimeError(f"{hackathon_path} field 'needs_gpu' must be a boolean")

    needs_gpu = cfg["needs_gpu"]
    if not needs_gpu:
        LOGGER.info("Device resolved | device=cpu (hackathon.json needs_gpu=false)")
        return "cpu"

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"hackathon.json requires GPU, but torch is unavailable: {exc}")
    if not torch.cuda.is_available():
        raise RuntimeError("hackathon.json requires GPU, but CUDA is not available.")

    gpu_name = torch.cuda.get_device_name(0)
    LOGGER.info("Device resolved | device=cuda gpu=%s (hackathon.json needs_gpu=true)", gpu_name)
    return "cuda"
