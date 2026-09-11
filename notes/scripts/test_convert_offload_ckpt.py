"""End-to-end test of convert_offload_ckpt on a small model: train with CPUOffloadAdamW, convert its
state, continue with torch.optim.AdamW, and verify the weights match a reference run that used
torch.optim.AdamW throughout."""
import sys, copy, torch
sys.path.insert(0, ".")
from mof.common.cpu_offload_optimizer import CPUOffloadAdamW
from notes.scripts.convert_offload_ckpt import convert_payload
from omegaconf import OmegaConf

torch.manual_seed(0)
def make(): return torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.Mish(), torch.nn.Linear(128, 8))
ref, off = make(), None
off = copy.deepcopy(ref)
kw = dict(lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
o_ref, o_off = torch.optim.AdamW(ref.parameters(), **kw), CPUOffloadAdamW(off.parameters(), **kw)
X, Y = torch.randn(40, 16, 32), torch.randn(40, 16, 8)
for i in range(20):  # phase 1: both optimizers
    torch.nn.functional.mse_loss(ref(X[i]), Y[i]).backward(); o_ref.step(); o_ref.zero_grad()
    torch.nn.functional.mse_loss(off(X[i]), Y[i]).backward(); o_off.step(); o_off.zero_grad()

payload = {"cfg": OmegaConf.create({"optimizer": {"_target_": "mof.common.cpu_offload_optimizer.CPUOffloadAdamW", "num_threads": 16},
                                    "training": {"device": "cuda:0", "ema_device": "cpu"}}),
           "state_dicts": {"optimizer": o_off.state_dict()}, "pickles": {}}
payload = convert_payload(payload)
assert payload["cfg"].optimizer._target_ == "torch.optim.AdamW"
assert "ema_device" not in payload["cfg"].training and "num_threads" not in payload["cfg"].optimizer

conv = copy.deepcopy(off)                      # weights as saved in state_dicts['model']
o_conv = torch.optim.AdamW(conv.parameters(), **kw)
o_conv.load_state_dict(payload["state_dicts"]["optimizer"])
for i in range(20, 40):  # phase 2: reference keeps going, converted run continues from the state
    torch.nn.functional.mse_loss(ref(X[i]), Y[i]).backward(); o_ref.step(); o_ref.zero_grad()
    torch.nn.functional.mse_loss(conv(X[i]), Y[i]).backward(); o_conv.step(); o_conv.zero_grad()
d = max((p - q).abs().max().item() for p, q in zip(ref.parameters(), conv.parameters()))
print(f"max |w_reference - w_converted| after 20 further steps: {d:.3e}")
assert d < 1e-6, d
print("PASS")
