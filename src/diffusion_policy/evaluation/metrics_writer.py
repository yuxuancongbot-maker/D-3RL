"""Evaluation metric serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    import wandb
except Exception:  # pragma: no cover - wandb may be unavailable in minimal envs
    wandb = None


def to_jsonable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert runner metrics into JSON-serializable values."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if wandb is not None:
            video_cls = getattr(getattr(wandb, "sdk", None), "data_types", None)
            video_type = getattr(video_cls, "video", None)
            video_class = getattr(video_type, "Video", None) if video_type is not None else None
            if video_class is not None and isinstance(value, video_class):
                out[key] = value._path
                continue
        if hasattr(value, "item"):
            try:
                out[key] = value.item()
                continue
            except Exception:
                pass
        out[key] = value
    return out


def write_eval_log(metrics: Mapping[str, Any], output_dir: str | Path, filename: str = "eval_log.json") -> Path:
    """Write evaluation metrics as sorted JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable_metrics(metrics), f, indent=2, sort_keys=True)
    return path
