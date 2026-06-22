from __future__ import annotations

import json

from diffusion_policy.evaluation.metrics_writer import write_eval_log
from diffusion_policy.training.checkpointing import TopKCheckpointManager


def test_topk_accepts_slash_or_sanitized_keys(tmp_path):
    mgr = TopKCheckpointManager(
        save_dir=str(tmp_path),
        monitor_key="test/mean_score",
        mode="max",
        k=1,
        format_str="epoch={epoch:04d}-score={test_mean_score:.3f}.ckpt",
    )
    path = mgr.get_ckpt_path({"epoch": 1, "test/mean_score": 0.5})
    assert path.endswith("epoch=0001-score=0.500.ckpt")

    path2 = mgr.get_ckpt_path({"epoch": 2, "test_mean_score": 0.6})
    assert path2.endswith("epoch=0002-score=0.600.ckpt")


def test_write_eval_log(tmp_path):
    out = write_eval_log({"test/mean_score": 0.75}, tmp_path)
    data = json.loads(out.read_text())
    assert data["test/mean_score"] == 0.75
