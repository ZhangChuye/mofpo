"""Roll out a trained checkpoint once in BiGym with DEPTH + POINT CLOUDS enabled on the three
policy cameras (head, left_wrist, right_wrist) and save everything to one HDF5 file.

Recorded per step (T = episode length):
  obs/rgb_<cam>        (T,84,84,3) uint8     policy-resolution RGB (what the policy sees)
  obs/depth_<cam>      (T,84,84)   float32   metric depth [m] from MuJoCo (same camera/res)
  obs/pcd_<cam>        (T,1024,6)  float32   BiGym point cloud: world-frame XYZ + RGB in [0,1]
  hires/rgb_<cam>      (T,256,256,3) uint8   same cameras + 'front_far' third-person at 256x256
  hires/depth_<cam>    (T,256,256)  float32  metric depth [m] at 256x256
  cam/<cam>/pos, xmat  (T,3), (T,3,3)        camera pose in world (MuJoCo convention: looks -z, y up)
  obs/<lowdim key>     (T,d)                 proprioception, EE poses, head/base pose
  action               (T,20)      float32   executed absolute EE-pose action
  reward, task_success, done (T,)
  attrs: intrinsics fx,fy,cx,cy per camera/resolution, fovy, control_frequency, seed, outcome
The camera intrinsics follow BiGym's pointcloud_generator (pinhole from fovy; cx=(W-1)/2).

  python notes/scripts/rollout_record_depth.py --ckpt <ckpt> --out_dir <dir> [--seeds 100000 ...]
"""
import os, sys, argparse, pathlib, collections, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.getcwd())
import numpy as np, torch, dill, hydra, mujoco, h5py, cv2, imageio
from omegaconf import OmegaConf
from mof.env.bigym.bigym_image_wrapper import BigymImageWrapper
from mof.common.pytorch_util import dict_apply
from bigym.action_modes import JointPositionActionMode  # noqa (import side effects)
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.rby1 import RBY1
from bigym.envs.move_plates import MoveTwoPlates
from bigym.utils.pointcloud_generator import camera_intrinsics_from_fovy, get_camera_fovy_deg, resolve_camera_id

CAMS = ["head", "left_wrist", "right_wrist"]
HIRES_CAMS = CAMS + ["front_far"]

def make_env_with_depth(control_frequency=20, low_pass_freq_hz=10.0, resolution=(84, 84), init_perturb=False, runtime_hold_steps=0):
    """Same as mof.env.bigym.factory.make_rby1_move_two_plates_env, but with depth + pcd on."""
    return MoveTwoPlates(
        action_mode=RBY1CartesianActionModeWholeBody(block_until_reached=False, direct_mode=False,
            control_frequency=control_frequency, interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz, runtime_hold_steps=runtime_hold_steps),
        observation_config=ObservationConfig(cameras=[
            CameraConfig(c, rgb=True, depth=True, resolution=resolution, pcd=True, pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0)
            for c in CAMS]),
        control_frequency=control_frequency, render_mode=None, robot_cls=RBY1, init_perturb=init_perturb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100000 + i for i in range(8)], help="tried in order; the first SUCCESSFUL episode is saved (plus the first failure if none succeed)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--hires", type=int, default=256)
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
    assert cfg.task_name == "rby1_move_two_plates", cfg.task_name
    ws = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(out / "_ws")); ws.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    device = torch.device(a.device); policy.to(device); policy.eval()
    er = cfg.task.env_runner; n_obs, n_act, max_steps = int(er.n_obs_steps), int(er.n_action_steps), int(er.max_steps)
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    res = tuple(shape_meta["obs"]["head_image"]["shape"][1:])
    raw_env = make_env_with_depth(resolution=res)
    env = BigymImageWrapper(raw_env, shape_meta=shape_meta, render_obs_key=er.render_obs_key)
    model, data = raw_env._mojo.model, raw_env._mojo.data
    cam_ids = {c: resolve_camera_id(model, c) for c in HIRES_CAMS}
    assert all(v >= 0 for v in cam_ids.values()), cam_ids
    hires = mujoco.Renderer(model, a.hires, a.hires)

    def render_hires(cam):
        hires.update_scene(data, camera=cam_ids[cam]); hires.disable_depth_rendering(); rgb = hires.render().copy()
        hires.enable_depth_rendering(); depth = hires.render().copy(); hires.disable_depth_rendering()
        return rgb, depth.astype(np.float32)

    lowdim_keys = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") != "rgb"] + ["proprioception"]

    def run_episode(seed):
        torch.manual_seed(int(cfg.training.seed) + seed)  # reproducible DDIM sampling noise
        env.seed(seed); obs = env.reset(); policy.reset()
        hist = collections.deque([obs] * n_obs, maxlen=n_obs)
        rec = collections.defaultdict(list)
        def record(obs, action, reward, done, info):
            raw = env.last_obs
            for c in CAMS:
                rec[f"obs/rgb_{c}"].append(np.moveaxis(raw[f"rgb_{c}"], 0, -1).astype(np.uint8))
                rec[f"obs/depth_{c}"].append(np.asarray(raw[f"depth_{c}"], dtype=np.float32))
                rec[f"obs/pcd_{c}"].append(np.asarray(raw[f"pcd_{c}"], dtype=np.float32))
            for c in HIRES_CAMS:
                rgb, depth = render_hires(c); rec[f"hires/rgb_{c}"].append(rgb); rec[f"hires/depth_{c}"].append(depth)
                rec[f"cam/{c}/pos"].append(np.array(data.cam_xpos[cam_ids[c]], dtype=np.float32))
                rec[f"cam/{c}/xmat"].append(np.array(data.cam_xmat[cam_ids[c]], dtype=np.float32).reshape(3, 3))
            for k in lowdim_keys:
                rec[f"obs/{k}"].append(np.asarray(raw[k], dtype=np.float32))
            rec["action"].append(np.asarray(action, dtype=np.float32) if action is not None else np.full(20, np.nan, np.float32))
            rec["reward"].append(float(reward)); rec["done"].append(bool(done)); rec["task_success"].append(float(info.get("task_success", 0.0)))
        record(obs, None, 0.0, False, {})  # t = 0 observation (no action yet)
        step, done, success = 0, False, False
        while not done and step < max_steps:
            stacked = {k: np.stack([h[k] for h in hist], axis=0)[None] for k in obs}
            with torch.no_grad():
                act = policy.predict_action(dict_apply(stacked, lambda x: torch.as_tensor(x, device=device)))["action"][0].cpu().numpy()
            for i in range(n_act):
                obs, rew, done, info = env.step(act[i]); hist.append(obs); step += 1
                success = success or (rew > 0) or bool(info.get("task_success", 0))
                record(obs, act[i], rew, done, info)
                if done or step >= max_steps: break
        return {k: np.stack(v) for k, v in rec.items()}, success, step

    saved = []
    first_fail = None
    for seed in a.seeds:
        rec, success, steps = run_episode(seed)
        print(f"seed {seed}: {'SUCCESS' if success else 'fail'} after {steps} steps", flush=True)
        if success or (first_fail is None):
            tag = "success" if success else "fail"
            path = out / f"rby1_move_two_plates_seed{seed}_{tag}_depth.h5"
            with h5py.File(path, "w") as f:
                for k, v in rec.items():
                    f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                f.attrs.update(dict(task="rby1_move_two_plates", seed=seed, outcome=tag, n_steps=steps, control_frequency=int(raw_env.control_frequency),
                    checkpoint=str(a.ckpt), policy="MoF single-frame (right EE frame), 500 epochs, seed 0",
                    action_layout="[left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6), left_gripper, right_gripper], absolute world-frame EE targets",
                    depth_units="meters (MuJoCo depth rendering)", pcd_layout="(1024, 6) = world XYZ + RGB in [0,1], random subsample, max_dist 3 m",
                    camera_convention="MuJoCo: camera looks along -z, y up; pixel (u,v) -> x=(u-cx)*z/fx, y=-(v-cy)*z/fy, z=-depth in camera frame"))
                for c in HIRES_CAMS:
                    fovy = get_camera_fovy_deg(model, cam_ids[c])
                    for (w, h), name in (((res[1], res[0]), "policy_res"), ((a.hires, a.hires), "hires")):
                        fx, fy, cx, cy = camera_intrinsics_from_fovy(fovy, w, h)
                        f.attrs[f"intrinsics/{c}/{name}"] = json.dumps(dict(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, fovy_deg=fovy))
            saved.append(str(path)); print("saved", path, flush=True)
            # preview: mid-episode frame (rgb | depth colormap for the 3 cams, + third person) and a merged PLY point cloud
            t = steps // 2
            tiles = []
            for c in CAMS:
                d = rec[f"hires/depth_{c}"][t]; dn = np.clip(d / 3.0, 0, 1); dcol = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
                tiles.append(np.concatenate([rec[f"hires/rgb_{c}"][t], dcol], axis=0))
            d = rec["hires/depth_front_far"][t]; dn = np.clip(d / 4.0, 0, 1); dcol = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
            tiles.append(np.concatenate([rec["hires/rgb_front_far"][t], dcol], axis=0))
            imageio.imwrite(out / f"preview_seed{seed}_{tag}_t{t}.png", np.concatenate(tiles, axis=1))
            pts = np.concatenate([rec[f"obs/pcd_{c}"][t] for c in CAMS], axis=0)
            with open(out / f"pointcloud_seed{seed}_{tag}_t{t}.ply", "w") as fp:
                fp.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                for p in pts: fp.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[3]*255)} {int(p[4]*255)} {int(p[5]*255)}\n")
            if success: break
            first_fail = path
    print("saved files:", saved)

if __name__ == "__main__":
    main()
