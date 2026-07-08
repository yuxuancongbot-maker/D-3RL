"""
Standalone training for VAE with prev_action feedback.
Bypasses Hydra config issues.
"""
import sys, os, time, math
sys.path.insert(0, 'src')

import torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm

OmegaConf.register_new_resolver('eval', eval, replace=True)

# ── Config ──
DEVICE = 'cuda:0'
BATCH_SIZE = 256
NUM_EPOCHS = 3000
LR = 1e-4
SAVE_EVERY = 200
OUTPUT_DIR = 'outputs/vae_feedback'

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

print(f'Train samples: {len(wrapped)}, Val samples: {len(val_dataset)}')
print(f'Batch: obs={wrapped[0]["obs"].shape}, action={wrapped[0]["action"].shape}, prev_action={wrapped[0]["prev_action"].shape}')

# ── Model ──
from diffusion_policy.model.action_predictor.vae_action_predictor import VAEModel

model = VAEModel(
    action_dim=2, action_horizon=16, obs_dim=20, obs_horizon=2,
    latent_dim=32, layer=256, use_ema=True,
    prev_action_horizon=8,  # KEY: 8-step feedback window
)
model.to(DEVICE)
if model.ema is not None:
    model.ema.to(DEVICE)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print(f'Decoder input dim: {model.net._decoder_input_dim} (should be 40+32+16=88)')

# ── Optimizer ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR*0.01)

os.makedirs(OUTPUT_DIR, exist_ok=True)
best_val_loss = float('inf')

# ── Training ──
for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    model.net.train()
    train_losses = []
    for batch in train_loader:
        # Normalize (VAEModel.get_loss expects normalized data)
        nbatch = normalizer.normalize(batch)
        obs = nbatch['obs'].to(DEVICE).float()
        action = nbatch['action'].to(DEVICE).float()
        prev_action = nbatch.get('prev_action', None)
        if prev_action is not None:
            prev_action = prev_action.to(DEVICE).float()

        loss, loss_info = model.get_loss(
            {'obs': obs[:, :2], 'action': action, 'prev_action': prev_action},
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
    avg_train_loss = np.mean(train_losses)

    # Validate
    if epoch % 50 == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                nbatch = normalizer.normalize(batch)
                obs = nbatch['obs'].to(DEVICE).float()
                action = nbatch['action'].to(DEVICE).float()
                prev_action = nbatch.get('prev_action', None)
                if prev_action is not None:
                    prev_action = prev_action.to(DEVICE).float()
                loss, _ = model.get_loss(
                    {'obs': obs[:, :2], 'action': action, 'prev_action': prev_action},
                    loss_args={}, device=DEVICE,
                )
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch {epoch:4d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'ema': model.ema.state_dict() if model.ema else {},
                'normalizer': normalizer,
                'epoch': epoch,
                'val_loss': avg_val_loss,
            }, os.path.join(OUTPUT_DIR, 'best.pt'))
    elif epoch % 10 == 0:
        print(f'Epoch {epoch:4d}: train_loss={avg_train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}')

    # Save checkpoint
    if epoch % SAVE_EVERY == 0:
        torch.save({
            'model': model.state_dict(),
            'ema': model.ema.state_dict() if model.ema else {},
            'normalizer': normalizer,
            'epoch': epoch,
            'val_loss': avg_val_loss if epoch % 50 == 0 else None,
        }, os.path.join(OUTPUT_DIR, f'epoch_{epoch:04d}.pt'))

# Final save
torch.save({
    'model': model.state_dict(),
    'ema': model.ema.state_dict() if model.ema else {},
    'normalizer': normalizer,
    'epoch': NUM_EPOCHS,
}, os.path.join(OUTPUT_DIR, 'final.pt'))
print(f'\nDone! Best val_loss={best_val_loss:.4f}')
print(f'Models saved to {OUTPUT_DIR}/')
