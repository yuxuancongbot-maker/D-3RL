"""
Train VAE predictor with SDEdit-style noisy prev_action.
prev_action + noise → VAE denoise → clean action (conditioned on obs).

Training: add random noise to prev_action, VAE learns to recover clean action.
Inference: add noise to VAE's own previous prediction, VAE denoises it.
This makes the VAE robust to its own noisy feedback.
"""
import sys, os
sys.path.insert(0, 'src')

import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra

OmegaConf.register_new_resolver('eval', eval, replace=True)

DEVICE = 'cuda:0'
BATCH_SIZE = 256
NUM_EPOCHS = 2000
LR = 1e-4
SAVE_EVERY = 200
OUTPUT_DIR = 'outputs/vae_sdedit'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ──
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

# ── Model ──
from diffusion_policy.model.action_predictor.vae_action_predictor import VAEModel

model = VAEModel(action_dim=2, action_horizon=16, obs_dim=20, obs_horizon=2,
                 latent_dim=32, layer=256, use_ema=True, prev_action_horizon=8)
model.to(DEVICE)
if model.ema:
    model.ema.to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ── Noise schedule for SDEdit ──
def add_noise(x, noise_level):
    """Add Gaussian noise at given level (0=clean, 1=pure noise)."""
    if noise_level <= 0:
        return x
    noise = torch.randn_like(x)
    return x + noise_level * noise

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

best_val_loss = float('inf')

# ── Training ──
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    model.net.train()
    train_losses = []
    for batch in train_loader:
        nbatch = normalizer.normalize(batch)
        obs = nbatch['obs'].to(DEVICE).float()[:, :2]  # [B, 2, 20]
        action = nbatch['action'].to(DEVICE).float()   # [B, 16, 2]
        prev_action = nbatch['prev_action'].to(DEVICE).float()  # [B, 8, 2]

        # SDEdit: add random noise to prev_action
        noise_level = torch.rand(1).item() * 1.5  # 0 to 1.5 (over-noise for robustness)
        noisy_prev = add_noise(prev_action, noise_level)

        loss, info = model.get_loss(
            {'obs': obs, 'action': action, 'prev_action': noisy_prev},
            loss_args={}, device=DEVICE,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if model.ema is not None:
            model.ema.update()
        train_losses.append(loss.item())

    scheduler.step()
    avg_train = np.mean(train_losses)

    if epoch % 50 == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                nbatch = normalizer.normalize(batch)
                obs = nbatch['obs'].to(DEVICE).float()[:, :2]
                action = nbatch['action'].to(DEVICE).float()
                prev_action = nbatch['prev_action'].to(DEVICE).float()
                # Test at noise level 0.5 (representative of inference noise)
                noisy_prev = add_noise(prev_action, 0.5)
                loss, _ = model.get_loss(
                    {'obs': obs, 'action': action, 'prev_action': noisy_prev},
                    loss_args={}, device=DEVICE,
                )
                val_losses.append(loss.item())
        avg_val = np.mean(val_losses)
        print(f'Epoch {epoch:4d}: train={avg_train:.4f} val={avg_val:.4f} lr={scheduler.get_last_lr()[0]:.2e}')
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': avg_val},
                       os.path.join(OUTPUT_DIR, 'best.pt'))
    elif epoch % 20 == 0:
        print(f'Epoch {epoch:4d}: train={avg_train:.4f} lr={scheduler.get_last_lr()[0]:.2e}')

    if epoch % SAVE_EVERY == 0:
        torch.save({'model': model.state_dict(), 'epoch': epoch},
                   os.path.join(OUTPUT_DIR, f'epoch_{epoch:04d}.pt'))

torch.save({'model': model.state_dict(), 'epoch': NUM_EPOCHS},
           os.path.join(OUTPUT_DIR, 'final.pt'))
print(f'\nDone! Best val_loss={best_val_loss:.4f}')
