# MoF (Mixture of Frames Policy) reproduction — handoff to Skynet

Written 2026-09-11 14:20 EDT, status lines refreshed 2026-09-12 18:35 EDT, on the Alienware box (`chuye-Alienware-Aurora-R11`).
Everything below is verified on that machine unless marked *estimate*.

Companion documents in this repo:
- `notes/mof_rby1_settings.md` — what the **paper** does (verified against the PDF and the code).
- `notes/reproduction_report.md` — what was **run here**, with results.
- This file — how to pick the work up on another machine.

---

## 1. TL;DR — state of the world

| item | status |
|---|---|
| Environment (conda `mof` + BiGym fork + MuJoCo/EGL) | ✅ working, fully reproducible from the repo |
| BiGym RBY1 datasets (4 of 5 tasks) | ✅ downloaded, 15 GB; FlipCup (32.6 GB) **not** downloaded |
| Paper baseline run (single-frame *Right*, MoveTwoPlates, seed 0, 500 epochs) | ✅ done, evaluated: **38.8 %** vs paper 47.2 ± 1.7 % |
| Rollout videos of that policy (10 episodes) | ✅ `data/outputs/sf_right_move_two_plates_seed0/videos/` |
| Depth + point-cloud rollout dataset (1 episode) | ✅ `…/depth_rollout/*.h5` |
| **MoF-MoE run (the actual method)** | ✅ done: 500 epochs in 53 h (2026-09-10 08:25 → 2026-09-12 13:22), evaluated: **56.0 %** vs paper 51.6 ± 4.4 % |
| MoF-MoE evaluation + videos | ✅ 5 checkpoints × 50 episodes on CPU (61 min each); videos in `…/mof_moe_move_two_plates_seed0_offload/videos/` |
| Released checkpoints from the authors | ❌ **none exist** — everything must be trained |

### ⚠️ Blocker on the current machine (2026-09-11)
The host's NVIDIA **userspace libraries were upgraded to 580.178.04 while the running kernel module
is still 580.173.02**. Consequence: the *running* training process is unaffected (it holds its CUDA
context and is still progressing), but **any new CUDA process fails** with
`CUDA error 804: forward compatibility was attempted on non supported HW`.

EGL rendering still works (checked 2026-09-12), so `chain3_after_mof_moe.sh` detects the missing CUDA
and falls back to running the **policy on the CPU** with 6 env workers; the rollouts are the same
seeds and the same simulator, only the network arithmetic moves to the CPU, so the success numbers
are comparable (slower, and not bitwise identical to a GPU pass). A reboot (`sudo reboot`, then check
`nvidia-smi`) restores GPU evaluation, and the script can simply be re-run afterwards. This is one
more reason to move.

---

## 2. Repository

| | |
|---|---|
| Fork (yours, push access) | `git@github.com:ZhangChuye/mofpo.git`, branch `main` |
| Upstream (authors) | `https://github.com/pointW/mofpo`, last upstream commit `2a1e5de` "Update citation" |
| Local path here | `/home/chuye/Documents/mofpo` |
| Head at handoff | `9b747cf` (plus the commits added while finishing this doc) |
| Working tree | clean except untracked `mof.egg-info/` (build artifact, ignorable) |
| Paper | arXiv 2607.11884, project page `https://mofpo.github.io`, PDF mirror `https://dianwang.io/assets/mofpo.pdf` |
| Datasets | HuggingFace `dian-wang/mof-datasets` (public, no token needed) |

### Commits added on top of upstream

```
9b747cf Milestone log filter and post-MoF-MoE evaluation chain
00fbe3a MoF-MoE run progress and loss curves
fc73f31 Document the depth/point-cloud rollout dataset
48ae830 Baseline result (single-frame Right policy, MoveTwoPlates: 38.8% over 5 ckpts), depth-recording rollout script
26cef38 Offline eval script fixes, morning chain script, optimizer thread option
72500ed CPU-offload AdamW + EMA device option for training MoF-MoE on an 8 GB GPU
d32a54a Reproduction notes, helper scripts, and optional per-epoch checkpoint saving
```

### Changes to the authors' code (only 4 files; everything else is upstream)

1. **`mof/common/cpu_offload_optimizer.py` (new)** — `CPUOffloadAdamW`. AdamW whose fp32 master
   weights, moments and update live on the CPU; per-parameter gradients are streamed to pinned CPU
   buffers during backward and freed on the GPU. It literally calls `torch.optim.AdamW` on the
   master weights, so the math is identical; verified bitwise-equal to `torch.optim.AdamW` including
   gradient accumulation and a changing learning rate (`notes/scripts/test_cpu_offload_optimizer.py`).
   **Needed only because the 8 GB card cannot hold MoF-MoE's optimizer state. On Skynet, drop it.**
2. **`mof/workspace/train_diffusion_unet_hybrid_workspace.py`** and
   **`…_async_eval_workspace.py`** —
   - `training.ema_device` (default = `training.device`): lets the EMA copy live on the CPU.
   - `checkpoint.save_epoch_ckpts_from=<epoch>`: additionally writes `epoch=XXXX.ckpt`
     (model + EMA, no optimizer, ~1 GB each) so the paper's "last 5 checkpoints" can be evaluated
     offline after training instead of during it.
   - EMA is stepped once per *optimizer* step (identical behaviour when `gradient_accumulate_every=1`,
     which is what the paper uses).
   - The action-MSE sampling block follows the policy's actual device.
3. **`mof/model/diffusion/ema_model.py`** — one-line: the EMA update moves the source tensor to the
   EMA parameter's device (no-op when both are on the GPU).

All three are backward compatible: **with no extra flags the code behaves exactly like upstream.**

### Helper scripts added (`notes/scripts/`, none of them touch upstream behaviour)

| script | purpose |
|---|---|
| `smoke_env.py` | Creates a BiGym RBY1 env, resets, steps demo actions, dumps camera frames + a third-person video. First thing to run on a new machine. |
| `gpu_mem_test.py` | Builds the real dataset + policy for a given config/batch size and reports parameter counts, peak GPU memory, ms/iteration and a 500-epoch estimate. Use it to pick the batch size on Skynet. |
| `mem_diag.py` | Traces GPU memory through one forward/backward of the policy. |
| `eval_ckpts.py` | **Paper-protocol evaluation.** Rebuilds policy + env runner from each checkpoint's own config, runs 50 held-out episodes (seeds 100000+), writes `eval_log.json` per checkpoint plus `summary.json`, and records the runner's head-camera videos. |
| `rollout_video.py` | Records composite videos: the three policy cameras on top, a 480×640 third-person MuJoCo camera below, outcome in the filename. |
| `rollout_record_depth.py` | Rolls out once with **depth + point clouds** enabled on all three cameras and saves one HDF5 (RGB, metric depth, XYZRGB clouds, camera poses/intrinsics, proprio, actions). |
| `convert_offload_ckpt.py` | Converts a `CPUOffloadAdamW` checkpoint into a plain `torch.optim.AdamW` one so a run started here can be **resumed on Skynet** with the standard optimizer. Verified exact (`test_convert_offload_ckpt.py`). |
| `epoch_milestones.py` | Filters `logs.json.txt` to one line per N epochs with rate/ETA. |
| `chain_after_baseline.sh`, `chain2_after_eval.sh`, `chain3_after_mof_moe.sh` | The unattended pipelines actually used: wait for training → evaluate → videos → launch the next run. |

---

## 3. Environment setup on Skynet

### 3.1 System prerequisites
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
A working NVIDIA driver with **EGL** (`/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`) is required for
fast rendering. Verify `nvidia-smi` and the kernel module match the userspace libs.

### 3.2 Python environment (~20 min, 9.7 GB)
```bash
git clone git@github.com:ZhangChuye/mofpo.git && cd mofpo
conda env create -f conda_environment.yaml     # or mamba
conda activate mof
bash install.sh                                # installs mof + robosuite/robomimic/dexmimicgen/mink/BiGym
pip install "setuptools<70"                    # imageio-ffmpeg needs pkg_resources; see gotchas
```
`install.sh` refuses to run outside the `mof` env. It clones the pinned forks into `./src/`
(`bigym` 2.6 GB, `dexmimicgen` 61 MB) as editable installs.

Verified versions here: Python 3.10.8, torch 2.1.0 (cu118), torchvision 0.16.0, numpy 1.26.4,
mujoco 3.3.5, diffusers 0.11.1, timm 1.0.22, hydra-core 1.2.0, zarr 2.12.0, robosuite 1.5.2,
robomimic 0.3.1, gymnasium 1.2.3, pytorch3d 0.7.5, BiGym 4.1.0 (fork `pointW/bigym@2658314`).

### 3.3 Smoke test (do this before launching anything long)
```bash
cd <repo> && PYTHONPATH=. MUJOCO_GL=egl python notes/scripts/smoke_env.py rby1_move_two_plates
```
Expect: `env created in ~6 s`, `step time ≈ 25 ms/step` with EGL, camera list containing
`rby1/head`, `rby1/left_wrist`, `rby1/right_wrist`, `rby1/front_far`, and PNG/MP4 files in
`notes/scripts/_smoke_out/`. If step time is ~450 ms you are on OSMesa (CPU) — fix `MUJOCO_GL=egl`.

### 3.4 Data
```bash
python -m mof.scripts.download_datasets --tasks bigym      # all 5 BiGym tasks (~47 GB)
python -m mof.scripts.download_datasets --tasks rby1_move_two_plates   # just this one (1.8 GB)
```
Downloaded here (15 GB total): `rby1_move_two_plates` 2.6 GB (incl. 896 MB zarr cache),
`rby1_store_kitchenware` 3.8 GB, `rby1_flip_sandwich` 2.3 GB, `rby1_dishwasher_load_plates` 6.3 GB.
**FlipCup was never downloaded** (32.6 GB, 224×224 frames; also the heaviest to train — it holds the
whole replay buffer in RAM). Re-downloading on Skynet is faster than copying from here.

The first training run per task builds `cache_rby1_100.zarr.zip` next to the demos (a few minutes);
subsequent runs load it in seconds. The cache is portable but regenerating is harmless.

---

## 4. What has been run, and the numbers

### 4.1 Paper targets (BiGym RBY1, success %, mean ± std. err. over 3 seeds)

| method | FlipCup | Move2Plates | Kitchenware | FlipSandwich | Dishwasher |
|---|---|---|---|---|---|
| MoF-MoE | 56.5 ± 1.4 | **51.6 ± 4.4** | 31.6 ± 2.4 | 40.0 ± 3.1 | 89.2 ± 3.9 |
| MoF-Ensemble | 59.2 ± 0.4 | 51.5 ± 1.5 | 33.7 ± 0.5 | 43.2 ± 2.0 | 89.7 ± 1.4 |
| Single-frame *Right* (Table 1) | 45.1 ± 1.9 | **47.2 ± 1.7** | 22.5 ± 0.7 | 34.3 ± 1.6 | 83.1 ± 0.5 |
| DP (world frame) | 17.2 ± 2.8 | 38.5 ± 0.4 | 22.0 ± 1.6 | 32.5 ± 3.1 | 61.5 ± 2.1 |

### 4.2 Run 1 — single-frame *Right* on MoveTwoPlates (COMPLETE)
Exact paper config, seed 0, 100 demos, batch 128, 500 epochs, 7.9 h wall clock (00:03 → 07:54 on
2026-09-10). Evaluated with 50 held-out episodes per checkpoint:

| ep 460 | ep 470 | ep 480 | ep 490 | ep 499 (final) | **mean** | paper |
|---|---|---|---|---|---|---|
| 44 % | 38 % | 48 % | 30 % | 34 % | **38.8 %** | 47.2 ± 1.7 % |

One seed vs the paper's three; ~8 points low, with a 30–48 % spread across checkpoints that is
typical diffusion-policy variance on this task. Artifacts:
`data/outputs/sf_right_move_two_plates_seed0/{checkpoints,offline_eval,offline_eval_latest,videos,depth_rollout}`.
Rollout videos: 4/10 successes with the final checkpoint.

### 4.3 Run 2 — MoF-MoE on MoveTwoPlates (IN PROGRESS)
```
data/outputs/mof_moe_move_two_plates_seed0_offload/
```
Launched 2026-09-10 08:25 with (all on one line in `chain2_after_eval.sh`):
```bash
./run.sh train.py --config-name=train_mof_moe task=bigym_rby1_move_two_plates \
  _target_=mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace \
  ++training.enable_rollout=false ++checkpoint.save_epoch_ckpts_from=460 \
  optimizer._target_=mof.common.cpu_offload_optimizer.CPUOffloadAdamW ++optimizer.num_threads=16 \
  ++training.ema_device=cpu dataloader.batch_size=64 training.gradient_accumulate_every=2 \
  logging.mode=offline hydra.run.dir=data/outputs/mof_moe_move_two_plates_seed0_offload
```
Status at 2026-09-11 14:16: **epoch 281/500**, 6.3–6.4 min/epoch, train loss 0.0026, val loss ~0.08
(2 held-out demos, noisy), no errors, `latest.ckpt` 7.7 GB (model + EMA + offload optimizer state,
rewritten every 10 epochs). GPU only ~46 % busy — the CPU-side AdamW/EMA over 385 M parameters is the
bottleneck.

Deviations from the paper for this run, all deliberate and documented: batch 64 × 2 accumulation
(= paper's 128, same gradient up to fp rounding), CPU-offload optimizer (same math), EMA on CPU,
rollouts done offline after training instead of during, 1 seed instead of 3.

---

## 5. Why the CPU-offload optimizer exists (and how to get rid of it)

Measured on the RTX 2080 SUPER (8 GB):

| model | params | fp32 weights+grads+AdamW | + EMA copy | fits in 8 GB? |
|---|---|---|---|---|
| single-frame (1 U-Net expert) | 126.6 M | 2.0 GB | 2.5 GB | yes — batch 128 peaks at 6.7 GB |
| **MoF-MoE / MoF-Ensemble (4 experts)** | **384.8 M** | **6.2 GB** | **7.7 GB** | **no — OOM at any batch size** |

On Skynet with ≥16 GB VRAM, none of this is needed. **Use the authors' unmodified command.**

---

## 6. How to continue on Skynet

### 6.1 Option A (recommended) — retrain MoF-MoE the authors' way
Cleanest scientifically: no offload optimizer, and rollouts happen during training.
```bash
CUDA_VISIBLE_DEVICES=0 ./run_async_eval.sh train.py \
    --config-name=train_mof_moe task=bigym_rby1_move_two_plates
```
*Estimate*: ~10 h on a 24 GB card for 500 epochs; with `run_async_eval.sh` a second GPU context runs
50-episode rollouts every 10 epochs, so the paper number falls out of the run itself.
For 3 seeds add `training.seed=1` / `=2` (and different `hydra.run.dir`).

If you would rather keep evaluation offline (one GPU, less memory pressure), use `run.sh` plus
`++training.enable_rollout=false ++checkpoint.save_epoch_ckpts_from=460` and then section 6.3.

### 6.2 Option B — resume the 281-epoch run instead of restarting
Saves ~1 GPU-day. Copy `data/outputs/mof_moe_move_two_plates_seed0_offload/` (7.4 GB, the checkpoint
is the bulk) to Skynet, then convert the optimizer state and resume:
```bash
python notes/scripts/convert_offload_ckpt.py \
    --in  data/outputs/mof_moe_move_two_plates_seed0_offload/checkpoints/latest.ckpt \
    --out data/outputs/mof_moe_move_two_plates_seed0_offload/checkpoints/latest.ckpt.converted
mv …/latest.ckpt …/latest.ckpt.offload_backup && mv …/latest.ckpt.converted …/latest.ckpt

# resume: same hydra.run.dir, training.resume=True is already the default
./run.sh train.py --config-name=train_mof_moe task=bigym_rby1_move_two_plates \
  _target_=mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace \
  ++training.enable_rollout=false ++checkpoint.save_epoch_ckpts_from=460 \
  dataloader.batch_size=128 logging.mode=offline \
  hydra.run.dir=data/outputs/mof_moe_move_two_plates_seed0_offload
```
The converter rewrites the checkpoint's baked-in config (`optimizer._target_` → `torch.optim.AdamW`,
drops `ema_device`) and hands the inner AdamW state straight over — it *is* a valid AdamW state dict
over the same parameters in the same order. Note the batch size changes from 64×2 to 128 at the
resume point; if you want strict continuity keep `dataloader.batch_size=64
training.gradient_accumulate_every=2`.

### 6.3 Evaluate (paper protocol) and make videos
```bash
# 50 held-out episodes for every epoch=*.ckpt in the run, then the final checkpoint
python notes/scripts/eval_ckpts.py --run_dir <run_dir> --n_envs 8
python notes/scripts/eval_ckpts.py --run_dir <run_dir> --ckpts checkpoints/latest.ckpt \
       --n_envs 8 --out_dir <run_dir>/offline_eval_latest
# composite third-person videos
python notes/scripts/rollout_video.py --ckpt <run_dir>/checkpoints/latest.ckpt \
       --out_dir <run_dir>/videos --seeds 100000 100001 100002 100003 100004
```
Always with `MUJOCO_GL=egl`. Bump `--n_envs` on a bigger box: each env worker costs ~1.9 GB RAM and
~130 MiB VRAM; 8 workers evaluate 50 episodes of MoveTwoPlates in ~3.5 min.
`notes/scripts/chain3_after_mof_moe.sh` does all of the above in one shot.

### 6.4 The remaining tasks
Each `task=` is self-contained; nothing else changes:
`bigym_rby1_move_two_plates`, `bigym_rby1_store_kitchenware`, `bigym_rby1_flip_sandwich`,
`bigym_rby1_dishwasher_load_plates`, `bigym_rby1_flip_cup`.
Note the max episode lengths differ (400 / 800 / 800 / 1300 / 400 steps), so evaluation on
Dishwasher and Kitchenware is 2–3× slower than MoveTwoPlates. FlipCup additionally needs its 32.6 GB
224×224 dataset and a lot of RAM (the replay buffer is loaded in memory).

### 6.5 Depth / point-cloud data
```bash
python notes/scripts/rollout_record_depth.py --ckpt <ckpt> --out_dir <dir> --seeds 100000 100001 …
```
Records the first successful episode with `depth=True, pcd=True` on all three cameras and writes one
HDF5: `obs/rgb_<cam>` (T,84,84,3), `obs/depth_<cam>` (T,84,84) float32 **metres**, `obs/pcd_<cam>`
(T,1024,6) world XYZ+RGB, plus 256×256 RGB/depth for the three cameras and the third-person camera,
per-step camera poses, intrinsics in the file attrs (pinhole from fovy 90°; at 84 px fx=fy=42,
cx=cy=41.5), proprioception, 20-D actions, rewards and success flags. MuJoCo camera convention
(looks along −z, y up). The released datasets contain **RGB only** — depth/pcd must be re-rendered
like this, and the released demos cannot be replayed for depth because they carry no reset state.

---

## 7. Gotchas discovered the hard way

1. **`mof/` has no `__init__.py`.** Run everything from the repo root; scripts outside the root need
   `PYTHONPATH=.`. `train.py` works because of its own path setup.
2. **Rollout workers are spawned processes that re-import the launching script** →
   any script that builds an env runner needs an `if __name__ == "__main__":` guard, or it forks
   forever. (`eval_ckpts.py` deadlocked for 10 minutes before this was found.)
3. **Do not `import train` in a helper script that builds env workers** — `train.py` reopens
   stdout/stderr at import time and the re-import in each worker hangs the handshake.
4. **OSMesa deadlocks inside those workers** and is ~20× slower. Always `MUJOCO_GL=egl`.
   EGL costs ~130 MiB VRAM per env process.
5. **EGL teardown noise**: every worker prints `EGLError` tracebacks from `Renderer.__del__` when it
   exits. Harmless; results are written before it.
6. **RAM is the evaluation limit**, not VRAM: ~1.9 GB per env worker. Two 8-env evaluations at once
   OOM-killed a job here. Run evaluations sequentially unless the box is big.
7. **`setuptools>=70` breaks `imageio-ffmpeg`** (`No module named 'pkg_resources'`), so mp4 writing
   fails. `pip install "setuptools<70"`.
8. **H.264 needs even frame dimensions** — composite video tiles must add up to even width/height.
9. **Epoch numbering is 0-based**: the paper's "epoch 500" is `latest.ckpt` after epoch 499, and
   `checkpoint.save_epoch_ckpts_from=460` yields 460/470/480/490 + `latest.ckpt` = the paper's five.
10. **`training.resume=True` is the default**: relaunching the same command with the same
    `hydra.run.dir` resumes from `latest.ckpt` (saved every 10 epochs), so a crash costs ≤10 epochs.
11. **W&B runs offline** here (`logging.mode=offline`); sync later with
    `wandb sync data/outputs/<run>/wandb/offline-run-*` if you want the dashboards.
12. **Driver/library mismatch** (section 1) — check `torch.cuda.is_available()` before starting any
    long job on a machine that has had unattended upgrades.

---

## 8. What to copy vs. re-create

| item | size | copy? |
|---|---|---|
| The repo (code + notes + scripts) | ~10 MB | `git clone` from the fork — everything is pushed |
| `data/bigym/*` demos | 15 GB | **re-download** from HuggingFace (faster than scp) |
| `data/outputs/mof_moe_move_two_plates_seed0_offload/` | 7.4 GB | copy only if resuming (option B) |
| `data/outputs/sf_right_move_two_plates_seed0/checkpoints/` | 6.0 GB | optional; the results are already recorded |
| `…/sf_right_move_two_plates_seed0/{videos,offline_eval*,depth_rollout}` | ~300 MB | worth copying — these are finished deliverables |
| conda env | 9.7 GB | **re-create** from `conda_environment.yaml` + `install.sh` |

Suggested: `rsync -av --progress data/outputs/sf_right_move_two_plates_seed0/{videos,offline_eval,offline_eval_latest,depth_rollout} skynet:<path>/`

---

## 9. Open items / next steps

1. ~~Finish MoF-MoE on MoveTwoPlates and evaluate it~~ — **done: 56.0 %** (paper 51.6 ± 4.4 %),
   +17.2 points over the single-frame baseline trained identically here.
2. **More seeds.** The paper averages 3 seeds; everything here is seed 0. The 30–48 % checkpoint
   spread on the baseline says single-seed numbers are worth ±5 points at least.
3. **The other four BiGym tasks** — same command, different `task=`. Dishwasher is the paper's
   easiest (89 %) and the most convincing quick win; FlipCup needs the big dataset.
4. **MoF-Ensemble** (`--config-name=train_mof_ensemble`) is the same cost as MoF-MoE and is the
   paper's second-best method; it is the cheapest ablation to add.
5. Optional: `train_dp` (world-frame Diffusion Policy) as the bottom reference — it is the 126 M
   single-expert size, so it trains in ~8 h even on the old box.
