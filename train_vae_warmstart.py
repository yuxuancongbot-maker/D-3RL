"""
Train VAE with prev_action feedback, initialized from old VAE weights.
"""
import sys, os
sys.path.insert(0, 'src')

import torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra, dill
from tqdm import tqdm

OmegaConf.register_new_resolver('eval', eval, replace=True)

DEVICE = 'cuda:0'
BATCH_SIZE = 256
NUM_EPOCHS = 1000
LR = 3e-5  # Lower LR for fine-tuning
SAVE_EVERY = 100
OUTPUT_DIR = 'outputs/vae_feedback_warm'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset & Normalizer ──
from diffusion_policy.policy.action_predictor_lowdim_policy import ActionPredictorDatasetWrapper

ds_cfg = OmegaConf.create({
    '_target_': 'diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset',
    'zarr_path': 'data/pusht/pusht_cchi_v7_replay.zarr',
    'horizon': 16, 'pad_before': 1, 'pad_after': 7,
    'seed': 42, 'val_ratio': 0.02, 'max_train_episodes': 90,
})
dataset = hydra.utils.instantiate(ds_cfg)
wrapped = ActionPredictorDatasetWrapper(dataset, prev_action_horizon=8, n_action_steps=8)
normalizer = wrapped.get_normalizer()
val_dataset = wrapped.get_validation_dataset()

train_loader = DataLoader(wrapped, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Train: {len(wrapped)}, Val: {len(val_dataset)}')

# ── Create new VAE, then warm-start from old VAE weights ──
from diffusion_policy.model.action_predictor.vae_action_predictor import VAEModel

new_vae = VAEModel(action_dim=2, action_horizon=16, obs_dim=20, obs_horizon=2,
                   latent_dim=32, layer=256, use_ema=True, prev_action_horizon=8)
new_vae.to(DEVICE)

# Load old VAE weights
old_path = 'weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt'
old_payload = torch.load(old_path, map_location=DEVICE, pickle_module=dill)
old_sd = old_payload['state_dicts']['model']

# Warm-start: copy matching weights, zero-init prev_action columns
with torch.no_grad():
    for key in new_vae.state_dict():
        if key in old_sd:
            old_w = old_sd[key]
            new_w = new_vae.state_dict()[key]
            if old_w.shape == new_w.shape:
                new_vae.state_dict()[key].copy_(old_w)
            elif 'decoder_net.0' in key and old_w.dim() == 2:
                # decoder_net.0.weight: old=[256,72], new=[256,88]
                # Copy first 72 columns, leave last 16 as-is (random init)
                new_vae.state_dict()[key][:, :old_w.shape[1]].copy_(old_w)
                print(f'  Warm-start {key}: old{list(old_w.shape)} -> new{list(new_w.shape)}, '
                      f'copied first {old_w.shape[1]} cols')
            else:
                print(f'  Skip {key}: old{list(old_w.shape)} vs new{list(new_w.shape)}')
    # encoder is identical — copy encoder_mean, encoder_logstd too
    for key in ['encoder_mean.weight', 'encoder_mean.bias',
                'encoder_logstd.weight', 'encoder_logstd.bias']:
        k = f'net.{key}'
        if k in old_sd:
            new_vae.state_dict()[k].copy_(old_sd[k])

print(f'Model params: {sum(p.numel() for p in new_vae.parameters()):,}')

if new_vae.ema is not None:
    new_vae.ema.to(DEVICE)

optimizer = torch.optim.AdamW(new_vae.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR*0.01)

best_val_loss = float('inf')

# ── Training ──
for epoch in range(1, NUM_EPOCHS + 1):
    new_vae.train()
    new_vae.net.train()
    train_losses = []
    for batch in train_loader:
        nbatch = normalizer.normalize(batch)
        obs = nbatch['obs'].to(DEVICE).float()
        action = nbatch['action'].to(DEVICE).float()
        prev_action = nbatch.get('prev_action')
        if prev_action is not None:
            prev_action = prev_action.to(DEVICE).float()

        loss, info = new_vae.get_loss(
            {'obs': obs[:, :2], 'action': action, 'prev_action': prev_action},
            loss_args={}, device=DEVICE,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(new_vae.parameters(), 1.0)
        optimizer.step()
        if new_vae.ema is not None:
            new_vae.ema.update()
        train_losses.append(loss.item())

    scheduler.step()
    avg_train_loss = np.mean(train_losses)

    if epoch % 50 == 0:
        new_vae.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                nbatch = normalizer.normalize(batch)
                obs = nbatch['obs'].to(DEVICE).float()
                action = nbatch['action'].to(DEVICE).float()
                prev_action = nbatch.get('prev_action')
                if prev_action is not None:
                    prev_action = prev_action.to(DEVICE).float()
                loss, _ = new_vae.get_loss(
                    {'obs': obs[:, :2], 'action': action, 'prev_action': prev_action},
                    loss_args={}, device=DEVICE,
                )
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch {epoch:4d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model': new_vae.state_dict(), 'epoch': epoch, 'val_loss': avg_val_loss},
                       os.path.join(OUTPUT_DIR, 'best.pt'))
    elif epoch % 20 == 0:
        print(f'Epoch {epoch:4d}: train={avg_train_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}')

    if epoch % SAVE_EVERY == 0:
        torch.save({'model': new_vae.state_dict(), 'epoch': epoch},
                   os.path.join(OUTPUT_DIR, f'epoch_{epoch:04d}.pt'))

torch.save({'model': new_vae.state_dict(), 'epoch': NUM_EPOCHS},
           os.path.join(OUTPUT_DIR, 'final.pt'))
print(f'\nDone! Best val_loss={best_val_loss:.4f}')
