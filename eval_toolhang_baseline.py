"""Quick baseline eval for Tool Hang: VAE k=0 and DDPM k=5."""
import sys; sys.path.insert(0,'src')
import torch, dill, numpy as np
from omegaconf import OmegaConf; import hydra
OmegaConf.register_new_resolver('eval',eval,replace=True)
device='cuda:0'

# Dataset normalizer
ds_cfg=OmegaConf.create({'_target_':'diffusion_policy.dataset.robomimic_replay_lowdim_dataset.RobomimicReplayLowdimDataset','dataset_path':'data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5','horizon':16,'pad_before':1,'pad_after':7,'obs_keys':['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'],'abs_action':True,'use_legacy_normalizer':False,'rotation_rep':'rotation_6d','seed':42,'val_ratio':0.02})
ds=hydra.utils.instantiate(ds_cfg); nm=ds.get_normalizer(); nm.to(device)

# VAE
p=torch.load('weights/lowdim/predictor/vae/vae_tool_hang_lowdim/checkpoints/latest.ckpt',map_location=device,pickle_module=dill)
oc=p['cfg']; OmegaConf.set_struct(oc.policy,False)
oc.policy.backend='vae'
oc.policy.model={'_target_':'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel','action_dim':10,'action_horizon':16,'obs_dim':53,'obs_horizon':2,'latent_dim':32,'layer':256,'use_ema':True,'pretrain':False,'ckpt_path':None,'prev_action_horizon':0}
vae=hydra.utils.instantiate(oc.policy); vae.to(device)
vae.load_state_dict(p['state_dicts']['model'],strict=False); vae.to(device); vae.eval(); vae.set_normalizer(nm)

# DDPM
p2=torch.load('weights/lowdim/diffusion/robomimic/cnn/tool_hang/epoch=0850-test_mean_score=0.818.ckpt',map_location=device,pickle_module=dill)
ref=hydra.utils.instantiate(p2['cfg'].policy); ref.to(device)
ref.load_state_dict(p2['state_dicts']['model'],strict=False); ref.to(device); ref.eval(); ref.set_normalizer(nm)

# Runner
runner_cfg=OmegaConf.create({'_target_':'diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner','output_dir':'/tmp/th_eval','dataset_path':'data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5','obs_keys':['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'],'n_train':0,'n_train_vis':0,'train_start_idx':0,'n_test':50,'n_test_vis':0,'test_start_seed':100000,'max_steps':700,'n_obs_steps':2,'n_action_steps':8,'n_latency_steps':0,'render_hw':[128,128],'fps':10,'crf':22,'past_action':False,'abs_action':True,'n_envs':6})
runner=hydra.utils.instantiate(runner_cfg)

from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
class FK(torch.nn.Module):
    def __init__(self,k,rs): super().__init__(); self.k=k; self.ki=rs.index(k)
    def select_action(self,obs,init_action,deterministic=True):
        B=obs.shape[0]; dev=obs.device
        return (torch.full((B,),self.k,dtype=torch.long,device=dev),torch.full((B,),self.ki,dtype=torch.long,device=dev),torch.zeros(B,device=dev),torch.zeros(B,device=dev))
rs=[0,5]

for k_val,label in [(0,'VAE k=0'),(5,'DDPM k=5')]:
    pol=AdaBridgerPolicy(source_policy=vae,refinement_policy=ref,scheduler=FK(k_val,rs).to(device),horizon=16,obs_dim=53,action_dim=10,n_action_steps=8,n_obs_steps=2,refinement_steps=rs,max_refinement_steps=5)
    pol.set_normalizer(nm); pol.to(device); pol.eval()
    log=runner.run(pol)
    score=log.get('test/mean_score',0)
    print(f'{label}: test/mean_score={score:.3f}')
