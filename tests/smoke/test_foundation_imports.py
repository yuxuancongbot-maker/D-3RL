from __future__ import annotations


def test_common_foundation_imports():
    from diffusion_policy.common.json_logger import JsonLogger
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import SequenceSampler
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    assert JsonLogger is not None
    assert dict_apply is not None
    assert ReplayBuffer is not None
    assert SequenceSampler is not None
    assert LinearNormalizer is not None


def test_model_foundation_imports():
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

    assert ConditionalUnet1D is not None
    assert LowdimMaskGenerator is not None
    assert MultiImageObsEncoder is not None
