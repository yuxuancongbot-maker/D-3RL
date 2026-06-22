from __future__ import annotations

import torch

from diffusion_policy.d3rl.refinement.ddim_warm_start import DDIMWarmStartRefiner
from diffusion_policy.d3rl.scheduler.cost_schedule import linear_cost_warmup, refinement_step_cost
from diffusion_policy.d3rl.scheduler.guards import SuccessGuard
from diffusion_policy.d3rl.scheduler.model import D3RLScheduler


def test_scheduler_outputs_valid_refinement_steps():
    scheduler = D3RLScheduler(
        obs_dim=3,
        action_dim=2,
        action_horizon=4,
        n_obs_steps=2,
        hidden_dim=16,
        refinement_steps=[0, 2, 5, 10],
    )
    obs = torch.zeros(5, 2, 3)
    init_action = torch.zeros(5, 4, 2)
    steps, action_idx, log_prob, value = scheduler.select_action(obs, init_action, deterministic=True)

    assert steps.shape == (5,)
    assert action_idx.shape == (5,)
    assert log_prob.shape == (5,)
    assert value.shape == (5,)
    assert set(steps.tolist()).issubset({0, 2, 5, 10})


def test_cost_warmup_and_step_cost():
    assert linear_cost_warmup(epoch=0, target=0.05, task_only_epochs=10, warmup_epochs=20) == 0.0
    assert linear_cost_warmup(epoch=10, target=0.05, task_only_epochs=10, warmup_epochs=20) == 0.0
    assert linear_cost_warmup(epoch=30, target=0.05, task_only_epochs=10, warmup_epochs=20) == 0.05
    assert refinement_step_cost(5, max_refinement_steps=10) == 0.5


def test_success_guard():
    guard = SuccessGuard(baseline_success=0.8, tolerance=0.05)
    assert not guard.is_violation(0.76)
    assert guard.is_violation(0.74)


def test_warm_start_zero_steps_returns_init_action():
    refiner = DDIMWarmStartRefiner(refinement_policy=object())
    init_action = torch.ones(2, 4, 3)
    out = refiner.refine(obs_dict={}, init_action=init_action, refinement_steps=0)
    assert out is init_action
