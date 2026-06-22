from __future__ import annotations


def test_package_imports():
    import diffusion_policy

    assert diffusion_policy.__version__


def test_core_modules_import():
    from diffusion_policy.workspace.base import BaseWorkspace
    from diffusion_policy.workspace.legacy_aliases import resolve_legacy_target
    from diffusion_policy.evaluation.checkpoint_loader import load_payload

    assert BaseWorkspace is not None
    assert resolve_legacy_target("x") == "x"
    assert load_payload is not None
