"""Convert a checkpoint trained with CPUOffloadAdamW into a plain torch.optim.AdamW checkpoint.

Use this to move a run started on a small GPU (which needed the CPU-offload optimizer) onto a
machine with enough VRAM for the standard optimizer, and resume training there with no loss of
state. The conversion is exact: CPUOffloadAdamW *is* a torch.optim.AdamW over CPU master weights,
so its inner optimizer state_dict is already a valid AdamW state_dict over the same parameters in
the same order, and the model weights on device are copied from the master weights at the end of
every step.

    python notes/scripts/convert_offload_ckpt.py --in <ckpt> --out <ckpt> [--keep-ema-device]

Afterwards resume with the *unmodified* upstream command; the new checkpoint's baked-in cfg no
longer references CPUOffloadAdamW.
"""
import argparse, pathlib, torch, dill
from omegaconf import OmegaConf


def convert_payload(payload, keep_ema_device=False):
    sd = payload["state_dicts"]
    opt = sd.get("optimizer")
    if opt is None:
        raise SystemExit("checkpoint has no optimizer state (epoch=*.ckpt files are saved without it)")
    if "inner" not in opt:
        raise SystemExit("optimizer state is not a CPUOffloadAdamW state_dict; nothing to convert")
    sd["optimizer"] = opt["inner"]                    # already a torch.optim.AdamW state_dict
    cfg = payload["cfg"]
    OmegaConf.set_struct(cfg, False)
    cfg.optimizer._target_ = "torch.optim.AdamW"
    cfg.optimizer.pop("num_threads", None)
    cfg.optimizer.pop("pin_memory", None)
    if not keep_ema_device:
        cfg.training.pop("ema_device", None)          # EMA returns to training.device
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--keep-ema-device", action="store_true", help="leave training.ema_device as-is")
    a = ap.parse_args()
    payload = torch.load(open(a.src, "rb"), pickle_module=dill, map_location="cpu")
    payload = convert_payload(payload, a.keep_ema_device)
    pathlib.Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, open(a.dst, "wb"), pickle_module=dill)
    print(f"wrote {a.dst}\n  optimizer -> {payload['cfg'].optimizer._target_}"
          f"\n  ema_device -> {payload['cfg'].training.get('ema_device', payload['cfg'].training.device)}"
          f"\n  epoch {dill.loads(payload['pickles']['epoch'])}, global_step {dill.loads(payload['pickles']['global_step'])}")


if __name__ == "__main__":
    main()
