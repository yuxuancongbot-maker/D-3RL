from __future__ import annotations

from omegaconf import OmegaConf

from diffusion_policy.workspace.base import BaseWorkspace


class DummyWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.global_step = 7
        self.epoch = 3

    def run(self):
        return None


def test_base_workspace_checkpoint_round_trip(tmp_path):
    cfg = OmegaConf.create({"_target_": "tests.unit.test_checkpointing.DummyWorkspace"})
    ws = DummyWorkspace(cfg, output_dir=str(tmp_path))
    ckpt = ws.save_checkpoint(use_thread=False)

    loaded = DummyWorkspace(cfg, output_dir=str(tmp_path))
    loaded.global_step = 0
    loaded.epoch = 0
    payload = loaded.load_checkpoint(ckpt)

    assert "cfg" in payload
    assert "state_dicts" in payload
    assert "pickles" in payload
    assert loaded.global_step == 7
    assert loaded.epoch == 3
