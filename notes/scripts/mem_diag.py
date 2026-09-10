import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import hydra, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
import train
from mof.common.pytorch_util import dict_apply
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
with hydra.initialize(config_path="../../mof/config", version_base=None):
    cfg = hydra.compose(config_name="train_mof_moe", overrides=["task=bigym_rby1_move_two_plates"])
OmegaConf.resolve(cfg)
dataset = hydra.utils.instantiate(cfg.task.dataset)
policy = hydra.utils.instantiate(cfg.policy); policy.set_normalizer(dataset.get_normalizer())
dev = torch.device("cuda:0"); policy.to(dev)
mem = lambda: f"{torch.cuda.memory_allocated()/1e9:.2f} GB alloc / peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
print("after policy.to(cuda):", mem())
batch = next(iter(DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)))
batch = dict_apply(batch, lambda x: x.to(dev))
print("batch shapes:", {k: tuple(v.shape) for k, v in batch["obs"].items() if k.endswith("image")}, "action", tuple(batch["action"].shape))
# hook encoder to print input shapes
orig = policy.obs_encoder.forward
def hooked(obs_dict):
    print("  obs_encoder call:", {k: tuple(v.shape) for k, v in obs_dict.items() if v.ndim >= 4}, mem()); return orig(obs_dict)
policy.obs_encoder.forward = hooked
with torch.no_grad():
    loss = policy.compute_loss(batch); print("no_grad compute_loss ok:", mem())
torch.cuda.reset_peak_memory_stats()
loss = policy.compute_loss(batch); print("with grad, after forward:", mem())
loss.backward(); print("after backward:", mem())
