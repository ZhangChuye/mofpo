"""Smoke test: create a BiGym RBY1 env via the MoF factory with EGL rendering,
reset, replay a few demo actions, dump head/wrist/third-person frames, time steps."""
import os, sys, time, glob
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch, mujoco, imageio
from safetensors.numpy import load_file
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("mujoco", mujoco.__version__, "MUJOCO_GL", os.environ["MUJOCO_GL"])
import bigym
from mof.env.bigym import factory
task = sys.argv[1] if len(sys.argv) > 1 else "rby1_move_two_plates"
make = getattr(factory, f"make_{task}_env")
t0 = time.time(); env = make(resolution=(84, 84)); print(f"env created in {time.time()-t0:.1f}s; control_frequency={env.control_frequency}")
t0 = time.time(); obs, info = env.reset(seed=100000); print(f"reset in {time.time()-t0:.1f}s")
print("obs keys:", {k: tuple(np.asarray(v).shape) for k, v in obs.items()})
model, data = env._mojo.model, env._mojo.data
cams = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
print("cameras in model:", cams)
demo = load_file(sorted(glob.glob(f"data/bigym/{task}/*.safetensors"))[0])
acts = demo["info_demo_action"].astype(np.float32)
out = "notes/scripts/_smoke_out"; os.makedirs(out, exist_ok=True)
rend = mujoco.Renderer(model, 480, 640)
third = [c for c in cams if "front_far" in c]
frames = []
t0 = time.time(); n = 40
for t in range(n):
    obs, rew, term, trunc, info = env.step(acts[min(t, len(acts)-1)])
    if third:
        rend.update_scene(data, camera=third[0]); frames.append(rend.render().copy())
dt = (time.time() - t0) / n
print(f"step time {dt*1000:.1f} ms/step ({1/dt:.1f} Hz); reward={rew}, term={term}, trunc={trunc}, success={info.get('task_success')}")
imageio.imwrite(f"{out}/{task}_head.png", np.moveaxis(obs["rgb_head"], 0, -1))
imageio.imwrite(f"{out}/{task}_left_wrist.png", np.moveaxis(obs["rgb_left_wrist"], 0, -1))
imageio.imwrite(f"{out}/{task}_right_wrist.png", np.moveaxis(obs["rgb_right_wrist"], 0, -1))
if frames:
    imageio.imwrite(f"{out}/{task}_third_person.png", frames[-1])
    imageio.mimsave(f"{out}/{task}_third_person.mp4", frames, fps=env.control_frequency)
print("saved frames to", out)
env.close()
