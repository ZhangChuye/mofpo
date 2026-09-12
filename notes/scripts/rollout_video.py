"""Roll out a trained checkpoint in the BiGym RBY1 env and record a composite video:
top row = the three 84x84 policy cameras (head, left wrist, right wrist, upscaled),
bottom = a 480x640 third-person MuJoCo camera ('rby1/front_far'). One mp4 per episode,
with the episode outcome in the file name. Uses the same env factory, obs wrapper,
obs history (n_obs_steps) and action chunking (n_action_steps) as the paper's runner.

  python notes/scripts/rollout_video.py --ckpt <ckpt> --out_dir <dir> --seeds 100000 100001 ...
"""
import os, sys, argparse, pathlib, collections
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.getcwd())
import numpy as np, torch, dill, hydra, mujoco, cv2, imageio
from omegaconf import OmegaConf
import train  # registers resolvers
from mof.env.bigym.bigym_image_wrapper import BigymImageWrapper
from mof.env_runner.bigym_image_runner import create_env
from mof.common.pytorch_util import dict_apply

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[100000 + i for i in range(5)])
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--camera", default="rby1/front_far")
ap.add_argument("--max_steps", type=int, default=None)
a = ap.parse_args()
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
payload = torch.load(open(a.ckpt, "rb"), pickle_module=dill, map_location="cpu")
cfg = payload["cfg"]
OmegaConf.set_struct(cfg, False)
# Evaluation never steps the optimizer. Building it would allocate the CPU-offload
# optimizer's master/gradient buffers (~3 GB for MoF-MoE), so swap in a stock AdamW
# (state is allocated lazily) and skip loading the optimizer state.
if "optimizer" in cfg and str(cfg.optimizer.get("_target_", "")).endswith("CPUOffloadAdamW"):
    cfg.optimizer._target_ = "torch.optim.AdamW"
    cfg.optimizer.pop("num_threads", None)
    cfg.optimizer.pop("pin_memory", None)
cls = hydra.utils.get_class(cfg._target_)
ws = cls(cfg, output_dir=str(out / "_ws")); ws.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)
policy = ws.ema_model if cfg.training.use_ema else ws.model
device = torch.device(a.device); policy.to(device); policy.eval()
er = cfg.task.env_runner
n_obs, n_act = int(er.n_obs_steps), int(er.n_action_steps)
max_steps = a.max_steps or int(er.max_steps)
shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
raw_env = create_env(er.make_env_fn, use_pointcloud_obs=False, shape_meta=shape_meta, render_obs_key=er.render_obs_key)
env = BigymImageWrapper(raw_env, shape_meta=shape_meta, render_obs_key=er.render_obs_key)
fps = int(raw_env.control_frequency)
model, data = raw_env._mojo.model, raw_env._mojo.data
renderer = mujoco.Renderer(model, 480, 640)

def third_person():
    renderer.update_scene(data, camera=a.camera); return renderer.render().copy()

def compose(obs, tp, step, status):
    tiles = [cv2.resize(np.moveaxis((obs[k] * 255).astype(np.uint8), 0, -1), (200, 200), interpolation=cv2.INTER_NEAREST)
             for k in ("head_image", "left_wrist_image", "right_wrist_image")]
    top = np.concatenate(tiles, axis=1); top = cv2.copyMakeBorder(top, 0, 0, 0, 640 - top.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for i, name in enumerate(("head", "left wrist", "right wrist")):
        cv2.putText(top, name, (i * 200 + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    frame = np.concatenate([top, tp], axis=0)
    cv2.putText(frame, f"{cfg.task_name}  step {step}/{max_steps}  {status}", (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

results = {}
for seed in a.seeds:
    # Seed torch per episode so the DDIM sampling noise is reproducible. Note that these videos are
    # ONE sample of a stochastic policy: the same checkpoint and env seed can succeed here and fail
    # in eval_ckpts.py (which batches many envs in a single forward pass and therefore draws
    # different noise). Success *rates* must come from eval_ckpts.py, not from counting videos.
    torch.manual_seed(int(cfg.training.seed) + seed)
    env.seed(seed); obs = env.reset(); policy.reset()
    hist = collections.deque([obs] * n_obs, maxlen=n_obs)
    frames = [compose(obs, third_person(), 0, "")]
    step, done, success = 0, False, False
    while not done and step < max_steps:
        stacked = {k: np.stack([h[k] for h in hist], axis=0)[None] for k in obs}  # (1, To, ...)
        with torch.no_grad():
            act = policy.predict_action(dict_apply(stacked, lambda x: torch.as_tensor(x, device=device)))["action"][0].cpu().numpy()
        for i in range(n_act):
            obs, rew, done, info = env.step(act[i]); hist.append(obs); step += 1
            success = success or (rew > 0) or bool(info.get("task_success", 0))
            frames.append(compose(obs, third_person(), step, "SUCCESS" if success else ""))
            if done or step >= max_steps: break
    tag = "success" if success else "fail"
    path = out / f"{cfg.task_name}_seed{seed}_{tag}.mp4"
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
    results[seed] = success
    print(f"seed {seed}: {tag} after {step} steps -> {path}", flush=True)
print(f"success {sum(results.values())}/{len(results)}")
