"""Checkpoint management utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager as LegacyTopKCheckpointManager


class TopKCheckpointManager(LegacyTopKCheckpointManager):
    """Top-k checkpoint manager with slash-key fallback.

    The legacy implementation expects the exact monitor key to appear in the log
    dictionary. This wrapper also supports a sanitized key where ``/`` is
    replaced by ``_``.
    """

    def _get_value(self, data: Mapping[str, float]) -> float:
        if self.monitor_key in data:
            return data[self.monitor_key]
        sanitized = self.monitor_key.replace("/", "_")
        if sanitized in data:
            return data[sanitized]
        raise KeyError(
            f"Monitor key {self.monitor_key!r} not found. Available keys: {sorted(data.keys())}"
        )

    def get_ckpt_path(self, data: Mapping[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = self._get_value(data)
        fmt_data = dict(data)
        for key, val in list(data.items()):
            fmt_data[key.replace("/", "_")] = val
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**fmt_data))

        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == "max":
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None

        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(delete_path):
            os.remove(delete_path)
        return ckpt_path
