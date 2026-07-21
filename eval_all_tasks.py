#!/usr/bin/env python
"""
统一评测脚本：所有 lowdim 任务，每个 checkpoint 50 seeds
用法: python eval_all_tasks.py can    (单个任务)
      python eval_all_tasks.py all    (全部任务)
"""
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch, dill, numpy as np, time, glob
from omegaconf import OmegaConf; import hydra
OmegaConf.register_new_resolver('eval', eval, replace=True)
DEVICE = 'cuda:0'
from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace

TASK_CONFIG = {
    'pusht':       {'ckpt_dir': f'{ROOT}/outputs/pusht/checkpoints/epoch=*.ckpt',       'env_type': 'pusht', 'max_steps': 300},
    'lift':        {'ckpt_dir': f'{ROOT}/outputs/lift/checkpoints/epoch=*.ckpt',        'env_type': 'robo',  'dataset': 'lift',      'max_steps': 400},
    'can':         {'ckpt_dir': f'{ROOT}/outputs/can/checkpoints/epoch=*.ckpt',         'env_type': 'robo',  'dataset': 'can',       'max_steps': 400},
    'square':      {'ckpt_dir': f'{ROOT}/outputs/square/checkpoints/epoch=*.ckpt',      'env_type': 'robo',  'dataset': 'square',    'max_steps': 400},
    'transport':   {'ckpt_dir': f'{ROOT}/outputs/train_transport/checkpoints/epoch=*.ckpt','env_type': 'robo','dataset': 'transport','max_steps': 700},
    'tool_hang':   {'ckpt_dir': f'{ROOT}/outputs/train_toolhang/checkpoints/epoch=*.ckpt','env_type': 'robo','dataset': 'tool_hang','max_steps': 700},
}

def eval_pusht():
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    for ckpt in sorted(glob.glob(TASK_CONFIG['pusht']['ckpt_dir'])):
        p = torch.load(ckpt, map_location=DEVICE, pickle_module=dill)
        ws = TrainAdaBridgerWorkspace(p['cfg'])
        ws.policy.load_state_dict(p['state_dicts']['policy'], strict=False)
        ws.policy.to(DEVICE); ws.policy.eval(); ws.policy.scheduler_deterministic = True
        ws.policy.set_normalizer(ws.policy.source_policy.normalizer)
        ss, kk, ll = [], [], []
        for seed in range(100000, 100050):
            env = PushTKeypointsEnv()
            env = MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)
            env.seed(seed); torch.manual_seed(seed); obs = env.reset(); ws.policy.reset()
            done = False; st, ti, rw = [], [], []
            while not done:
                raw = obs[np.newaxis].astype(np.float32)
                if raw.shape[-1] == 40: raw = raw[..., :20]
                od = {'obs': torch.from_numpy(raw[:,:2]).float().to(DEVICE)}
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(): r = ws.policy.predict_action(od, return_intermediate=True)
                torch.cuda.synchronize(); t = (time.perf_counter() - t0) * 1000
                k = int(r['refinement_steps'].flatten()[0].item())
                st.append(k); ti.append(t)
                a = r['action'][0, :8].cpu().numpy()
                obs, reward, done, info = env.step(a)
                if reward is not None: rw.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))
            env.close()
            mr = float(np.max(rw)) if rw else 0.0
            ss.append(mr > 0.5); kk.append(np.mean(st)); ll.append(np.mean(ti))
        ep = ckpt.split('epoch=')[1].split('-')[0]
        print(f'PushT ep{ep}: succ={np.mean(ss):.1%} avg_k={np.mean(kk):.2f} lat={np.mean(ll):.1f}ms', flush=True)

def eval_robo(task_name):
    cfg = TASK_CONFIG[task_name]
    ds_path = f'{ROOT}/data/robomimic/datasets/{cfg["dataset"]}/ph/low_dim_abs.hdf5'
    rcfg = OmegaConf.create({'_target_': 'diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner',
        'output_dir': f'/tmp/eval_{task_name}', 'dataset_path': ds_path,
        'obs_keys': ['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'],
        'n_train':0,'n_train_vis':0,'train_start_idx':0,'n_test':1,'n_test_vis':0,'test_start_seed':100000,
        'max_steps':cfg['max_steps'],'n_obs_steps':2,'n_action_steps':8,'n_latency_steps':0,
        'render_hw':[128,128],'fps':10,'crf':22,'past_action':False,'abs_action':True,'n_envs':1})
    runner = hydra.utils.instantiate(rcfg)
    env = runner.env_fns[0]()
    for ckpt in sorted(glob.glob(cfg['ckpt_dir'])):
        p = torch.load(ckpt, map_location=DEVICE, pickle_module=dill)
        ws = TrainAdaBridgerWorkspace(p['cfg'])
        ws.policy.load_state_dict(p['state_dicts']['policy'], strict=False)
        ws.policy.to(DEVICE); ws.policy.eval(); ws.policy.scheduler_deterministic = True
        ws.policy.set_normalizer(ws.policy.source_policy.normalizer)
        ss, kk, ll = [], [], []
        for seed in range(100000, 100050):
            env.seed(seed); torch.manual_seed(seed); obs = env.reset(); ws.policy.reset()
            done = False; st, ti, rw = [], [], []
            while not done:
                od = {'obs': torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)}
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(): r = ws.policy.predict_action(od, return_intermediate=True)
                torch.cuda.synchronize(); t = (time.perf_counter() - t0) * 1000
                k = int(r['refinement_steps'].flatten()[0].item())
                st.append(k); ti.append(t)
                a = r['action'][0].cpu().numpy()
                a_env = runner.undo_transform_action(a)
                obs, reward, done, info = env.step(a_env)
                if isinstance(done, np.ndarray): done = np.all(done)
                if reward is not None: rw.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))
            mr = float(np.max(rw)) if rw else 0.0
            ss.append(mr > 0.5); kk.append(np.mean(st)); ll.append(np.mean(ti))
        ep = ckpt.split('epoch=')[1].split('-')[0]
        print(f'{task_name} ep{ep}: succ={np.mean(ss):.1%} avg_k={np.mean(kk):.2f} lat={np.mean(ll):.1f}ms', flush=True)
    env.close()

if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'all'
    tasks = ['pusht','lift','can','square','transport','tool_hang'] if task == 'all' else [task]
    for t in tasks:
        print(f'\n=== {t} ===', flush=True)
        if TASK_CONFIG[t]['env_type'] == 'pusht': eval_pusht()
        else: eval_robo(t)
    print('\nDone!', flush=True)
