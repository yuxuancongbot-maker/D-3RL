"""Square epoch 49: 3 runs with different seed groups"""
import sys, os
ROOT = '/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/icml_and_iros/d3rl_diffusion_policy'
sys.path.insert(0, ROOT + '/src')
import torch, dill, numpy as np, time
from omegaconf import OmegaConf; import hydra
OmegaConf.register_new_resolver('eval', eval, replace=True)
d = 'cuda:0'
from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace

ckpt = ROOT + '/outputs/square/checkpoints/epoch=0049-test_mean_score=0.920.ckpt'
p = torch.load(ckpt, map_location=d, pickle_module=dill)
rcfg = OmegaConf.create({'_target_':'diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner',
    'output_dir':'/tmp/ev_sq49', 'dataset_path': ROOT + '/data/robomimic/datasets/square/ph/low_dim_abs.hdf5',
    'obs_keys':['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'],
    'n_train':0,'n_train_vis':0,'train_start_idx':0,'n_test':1,'n_test_vis':0,'test_start_seed':100000,
    'max_steps':400,'n_obs_steps':2,'n_action_steps':8,'n_latency_steps':0,
    'render_hw':[128,128],'fps':10,'crf':22,'past_action':False,'abs_action':True,'n_envs':1})
runner = hydra.utils.instantiate(rcfg)
env = runner.env_fns[0]()

ss_all, kk_all, ll_all = [], [], []
for ri, start in enumerate([100000, 100050, 100100]):
    ws = TrainAdaBridgerWorkspace(p['cfg'])
    ws.policy.load_state_dict(p['state_dicts']['policy'], strict=False)
    ws.policy.to(d); ws.policy.eval(); ws.policy.scheduler_deterministic = True
    ws.policy.set_normalizer(ws.policy.source_policy.normalizer)
    ss, kk, ll = [], [], []
    for seed in range(start, start+50):
        env.seed(seed); torch.manual_seed(seed); obs = env.reset(); ws.policy.reset()
        done = False; st, ti, rw = [], [], []
        while not done:
            od = {'obs': torch.from_numpy(obs).float().unsqueeze(0).to(d)}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(): r = ws.policy.predict_action(od, return_intermediate=True)
            torch.cuda.synchronize(); t = (time.perf_counter() - t0) * 1000
            k = int(r['refinement_steps'].flatten()[0].item())
            st.append(k); ti.append(t)
            a = r['action'][0].cpu().numpy(); a_env = runner.undo_transform_action(a)
            obs, reward, done, info = env.step(a_env)
            if isinstance(done, np.ndarray): done = np.all(done)
            if reward is not None: rw.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))
        mr = float(np.max(rw)) if rw else 0.0
        ss.append(mr > 0.5); kk.append(np.mean(st)); ll.append(np.mean(ti))
    print(f'Square ep49 run{ri+1}: succ={np.mean(ss):.1%} avg_k={np.mean(kk):.2f} lat={np.mean(ll):.1f}ms', flush=True)
    ss_all.append(np.mean(ss)); kk_all.append(np.mean(kk)); ll_all.append(np.mean(ll))
env.close()
print(f'Square ep49 FINAL: succ={np.mean(ss_all):.1%}±{np.std(ss_all):.1%} avg_k={np.mean(kk_all):.2f}±{np.std(kk_all):.2f} lat={np.mean(ll_all):.1f}±{np.std(ll_all):.1f}ms', flush=True)
