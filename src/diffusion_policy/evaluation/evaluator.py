"""Checkpoint evaluation helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import torch

from diffusion_policy.evaluation.checkpoint_loader import (
    load_workspace_from_checkpoint,
    select_policy_from_workspace,
)
from diffusion_policy.evaluation.metrics_writer import write_eval_log


def run_checkpoint_eval(
    checkpoint: str | Path,
    output_dir: str | Path,
    device: str | torch.device = "cuda:0",
    use_ema: bool | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint, run its configured env runner, and write eval_log.json."""
    output_dir = Path(output_dir)
    workspace, payload = load_workspace_from_checkpoint(
        checkpoint,
        output_dir=str(output_dir),
        map_location=map_location,
    )
    cfg = payload["cfg"]
    policy = select_policy_from_workspace(workspace, use_ema=use_ema)
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    start = time.perf_counter()
    runner_log = env_runner.run(policy)
    total_wall_time = time.perf_counter() - start
    runner_log["eval/total_wall_time"] = total_wall_time
    write_eval_log(runner_log, output_dir)
    return runner_log
