"""Offline evaluation of saved checkpoints with the paper protocol (50 held-out episodes per
checkpoint), rebuilding policy + env runner from each checkpoint's own config, like eval.py.
Only n_envs / n_test_vis are overridable (they change wall-clock and how many videos are
recorded, not the per-episode results). Usage:

  python notes/scripts/eval_ckpts.py --run_dir data/outputs/<run> --ckpts checkpoints/epoch=0460.ckpt ... \
      --n_envs 8 --out_dir data/outputs/<run>/offline_eval
"""
import os, sys, json, pathlib, argparse, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.getcwd())
import torch, dill, hydra, wandb, numpy as np
from omegaconf import OmegaConf
import train  # registers hydra resolvers used by the configs
from mof.workspace.base_workspace import BaseWorkspace

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--ckpts", nargs="+", default=None, help="paths relative to run_dir (default: all checkpoints/epoch=*.ckpt)")
ap.add_argument("--out_dir", default=None)
ap.add_argument("--n_envs", type=int, default=8)
ap.add_argument("--n_test", type=int, default=None, help="default: value baked in the checkpoint config (paper: 50)")
ap.add_argument("--n_test_vis", type=int, default=None, help="default: record every test episode")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
run_dir = pathlib.Path(a.run_dir)
out_root = pathlib.Path(a.out_dir or run_dir / "offline_eval"); out_root.mkdir(parents=True, exist_ok=True)
ckpts = [run_dir / c for c in a.ckpts] if a.ckpts else sorted((run_dir / "checkpoints").glob("epoch=*.ckpt"))
assert ckpts, "no checkpoints found"
summary = {}
for ckpt in ckpts:
    tag = ckpt.stem
    out_dir = out_root / tag; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {tag} ===", flush=True)
    payload = torch.load(ckpt.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.set_struct(cfg, False)
    cfg.task.env_runner.n_envs = a.n_envs
    if a.n_test is not None: cfg.task.env_runner.n_test = a.n_test
    cfg.task.env_runner.n_test_vis = a.n_test_vis if a.n_test_vis is not None else cfg.task.env_runner.n_test
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(out_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(a.device)); policy.eval()
    t0 = time.time()
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out_dir))
    torch.manual_seed(int(cfg.training.seed)); np.random.seed(int(cfg.training.seed))
    runner_log = env_runner.run(policy)
    env_runner.env.close()
    log = {}
    for k, v in runner_log.items():
        log[k] = v._path if isinstance(v, wandb.sdk.data_types.video.Video) else (float(v) if isinstance(v, (np.floating, float, int)) else v)
    log["epoch"] = int(payload["pickles"] and dill.loads(payload["pickles"]["epoch"])) if "epoch" in payload["pickles"] else None
    log["eval_wall_time_s"] = time.time() - t0
    json.dump(log, (out_dir / "eval_log.json").open("w"), indent=2, sort_keys=True)
    summary[tag] = log["test/mean_score"]
    print(f"{tag}: test/mean_score = {log['test/mean_score']:.3f}  ({log['eval_wall_time_s']/60:.1f} min)", flush=True)
    del env_runner, workspace, policy, payload; torch.cuda.empty_cache()
scores = list(summary.values())
summary["mean_over_ckpts"] = float(np.mean(scores)); summary["n_ckpts"] = len(scores)
json.dump(summary, (out_root / "summary.json").open("w"), indent=2)
print("SUMMARY:", json.dumps(summary, indent=2))
