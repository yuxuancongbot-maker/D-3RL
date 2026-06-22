"""Unified Hydra training CLI."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.workspace.base import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _default_config_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3] / "configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D3RL Diffusion Policy training CLI")
    parser.add_argument("--config-name", default="config", help="Hydra config name")
    parser.add_argument("--config-dir", default=None, help="Hydra config directory")
    parser.add_argument("--output-dir", default=None, help="Training output directory")
    parser.add_argument("--cwd", default=None, help="Working directory for relative legacy paths")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides")
    return parser


def run(
    config_name: str,
    config_dir: str | None,
    output_dir: str | None,
    cwd: str | None,
    overrides: list[str],
) -> None:
    if cwd is not None:
        os.chdir(pathlib.Path(cwd).expanduser().resolve())
    config_dir_path = pathlib.Path(config_dir).resolve() if config_dir else _default_config_dir()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir_path)):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    output_dir_path = pathlib.Path(output_dir or "outputs/train").resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(output_dir_path))
    workspace.run()


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    run(args.config_name, args.config_dir, args.output_dir, args.cwd, args.overrides)


if __name__ == "__main__":
    main()
