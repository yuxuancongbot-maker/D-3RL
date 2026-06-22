"""Compatibility workspace alias for Action Predictor training."""

from diffusion_policy.workspace.train_action_predictor_lowdim_workspace import (
    TrainActionPredictorLowdimWorkspace,
)


class ActionPredictorWorkspace(TrainActionPredictorLowdimWorkspace):
    """Default Action Predictor workspace.

    The concrete low-dimensional workspace remains available at its legacy import
    path. This alias keeps newer configs that target ``ActionPredictorWorkspace``
    functional instead of routing them to a placeholder.
    """


__all__ = ["ActionPredictorWorkspace"]
