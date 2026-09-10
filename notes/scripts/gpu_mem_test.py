"""Instantiate the paper's MoF-MoE config on a task, build the real dataset (creates the cache),
count params, and measure GPU memory + time for training iterations at the paper batch size."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import hydra, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
import train  # registers resolvers
from mof.common.pytorch_util import dict_apply
task = sys.argv[1] if len(sys.argv) > 1 else "bigym_rby1_move_two_plates"
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 128
with hydra.initialize(config_path="../../mof/config", version_base=None):
    cfg = hydra.compose(config_name=os.environ.get("CFG","train_mof_moe"), overrides=[f"task={task}"] + os.environ.get("OVR","").split())
OmegaConf.resolve(cfg)
t0 = time.time(); dataset = hydra.utils.instantiate(cfg.task.dataset); print(f"dataset built in {time.time()-t0:.0f}s, len={len(dataset)}, n_episodes={dataset.replay_buffer.n_episodes}")
policy = hydra.utils.instantiate(cfg.policy); policy.set_normalizer(dataset.get_normalizer())
n = sum(p.numel() for p in policy.parameters()); n_enc = sum(p.numel() for p in policy.obs_encoder.parameters())
print(f"params: total {n/1e6:.1f}M, obs_encoder {n_enc/1e6:.1f}M, denoiser {(n-n_enc)/1e6:.1f}M")
dl = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
print(f"iters/epoch at batch {bs}: {len(dl)}")
device = torch.device("cuda:0"); policy.to(device)
if os.environ.get("GRAD_CKPT","0")=="1":
    n_ck=0
    for m in policy.obs_encoder.modules():
        if hasattr(m, "set_grad_checkpointing"): m.set_grad_checkpointing(True); n_ck+=1
    print("grad checkpointing enabled on", n_ck, "timm models")
opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
import copy; ema = copy.deepcopy(policy) if os.environ.get("EMA_ON_GPU","1")=="1" else None  # EMA copy on GPU unless offloaded
torch.cuda.reset_peak_memory_stats()
times = []
for i, batch in enumerate(dl):
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    torch.cuda.synchronize(); t0 = time.time()
    loss = policy.compute_loss(batch); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); times.append(time.time() - t0)
    if i == 0: print(f"first iter ok, loss={loss.item():.4f}")
    if i >= 20: break
print(f"iter time (steady): {sum(times[3:])/len(times[3:])*1000:.0f} ms; peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved")
print(f"estimated epoch time: {len(dl)*sum(times[3:])/len(times[3:])/60:.1f} min -> 500 epochs = {500*len(dl)*sum(times[3:])/len(times[3:])/3600:.1f} h (training only)")
