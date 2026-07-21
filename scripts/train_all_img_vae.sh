#!/bin/bash
ROOT=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/icml_and_iros/d3rl_diffusion_policy
cd "$ROOT" || exit 1

# --output-dir 必须放在 Hydra override 前面

tmux new -s img_pusht     -d "python -u -m diffusion_policy.cli.train --output-dir outputs/train_img_pusht     experiment=action_predictor_vae_image task=pusht_image      training=vae_image checkpoint=vae_val +encoder_output_dim=128 training.device=cuda:0 logging.mode=offline 2>&1 | tee /tmp/img_pusht.log"
tmux new -s img_lift      -d "python -u -m diffusion_policy.cli.train --output-dir outputs/train_img_lift      experiment=action_predictor_vae_image task=lift_image       training=vae_image checkpoint=vae_val +encoder_output_dim=128 training.device=cuda:1 logging.mode=offline 2>&1 | tee /tmp/img_lift.log"
tmux new -s img_can       -d "python -u -m diffusion_policy.cli.train --output-dir outputs/train_img_can       experiment=action_predictor_vae_image task=can_image        training=vae_image checkpoint=vae_val +encoder_output_dim=128 training.device=cuda:2 logging.mode=offline 2>&1 | tee /tmp/img_can.log"
tmux new -s img_transport -d "python -u -m diffusion_policy.cli.train --output-dir outputs/train_img_transport experiment=action_predictor_vae_image task=transport_image  training=vae_image checkpoint=vae_val +encoder_output_dim=128 training.device=cuda:3 logging.mode=offline 2>&1 | tee /tmp/img_transport.log"

echo "started: img_pusht(GPU0) img_lift(GPU1) img_can(GPU2) img_transport(GPU3)"
