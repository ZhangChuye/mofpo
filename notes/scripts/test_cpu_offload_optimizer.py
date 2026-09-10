"""Equivalence test: CPUOffloadAdamW vs torch.optim.AdamW on the same model/data (CPU-only
if no GPU is free). Both must produce the same weights (up to fp rounding) after N steps,
including with gradient accumulation and a changing learning rate."""
import sys, copy, torch
sys.path.insert(0, ".")
from mof.common.cpu_offload_optimizer import CPUOffloadAdamW
torch.manual_seed(0)
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
def make():
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GroupNorm(8, 256), torch.nn.Mish(),
                               torch.nn.Conv1d(1, 4, 5, padding=2), torch.nn.Flatten(), torch.nn.Linear(1024, 20))
ref = make(); off = copy.deepcopy(ref); ref.to(dev); off.to(dev)
kw = dict(lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
o_ref = torch.optim.AdamW(ref.parameters(), **kw); o_off = CPUOffloadAdamW(off.parameters(), num_threads=8, **kw)
accum, steps = 2, 60
x_all = torch.randn(steps * accum, 8, 64, device=dev); y_all = torch.randn(steps * accum, 8, 20, device=dev)
def model_loss(model, x, y):
    h = model[2](model[1](model[0](x)))            # (8, 256)
    h = model[3](h.unsqueeze(1))                   # (8, 4, 256)
    h = model[5](model[4](h))                      # (8, 20)
    return torch.nn.functional.mse_loss(h, y)
for s in range(steps):
    lr = 1e-4 * (0.5 + 0.5 * s / steps)
    for g in o_ref.param_groups: g["lr"] = lr
    for g in o_off.param_groups: g["lr"] = lr
    for a in range(accum):
        i = s * accum + a
        (model_loss(ref, x_all[i], y_all[i]) / accum).backward()
        (model_loss(off, x_all[i], y_all[i]) / accum).backward()
    o_ref.step(); o_ref.zero_grad(); o_off.step(); o_off.zero_grad()
    if dev != "cpu": torch.cuda.synchronize()
maxdiff = max((p - q).abs().max().item() for p, q in zip(ref.parameters(), off.parameters()))
print(f"max |w_ref - w_offload| after {steps} steps (accum={accum}): {maxdiff:.3e}")
# state_dict round trip
sd = o_off.state_dict(); o2 = CPUOffloadAdamW(copy.deepcopy(off).parameters(), **kw); o2.load_state_dict(sd)
print("state_dict round-trip ok:", all(torch.equal(a, b) for a, b in zip(o_off._master, o2._master)))
assert maxdiff < 1e-6, maxdiff
print("PASS")
