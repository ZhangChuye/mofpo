# MoF on BiGym RBY1 — reproduction log (single RTX 2080 SUPER, 8 GB)

Companion to [`mof_rby1_settings.md`](mof_rby1_settings.md) (what the paper does). This file records
what was actually run on this machine, with every deviation from the paper called out.

## Environment (verified)
- conda env `mof` from `conda_environment.yaml` + `install.sh` (torch 2.1.0/cu118, mujoco 3.3.5,
  BiGym fork `pointW/bigym@2658314`). System GL packages from the README installed via apt.
- Rendering: EGL on the NVIDIA driver (25 ms per env step incl. 3 cameras). OSMesa also works but
  is ~20x slower and deadlocks inside the forked rollout workers, so it is not used.
- Data: `dian-wang/mof-datasets`, BiGym tasks MoveTwoPlates (200 demos on disk, first 100 used),
  StoreKitchenware, FlipSandwich, DishwasherLoadPlates (100 each). FlipCup (224 px, 32 GB) not
  downloaded yet.
- Released checkpoints: none exist, so everything below is trained from scratch.

## Hard constraint: GPU memory
| model | params | fp32 weights+grads+AdamW | + EMA | fits on 8 GB? |
|---|---|---|---|---|
| single-frame policy (1 U-Net expert) | 126.6 M | 2.0 GB | 2.5 GB | yes, paper batch 128 (6.7 GB peak) |
| MoF-MoE / MoF-Ensemble (4 experts) | 384.8 M | 6.2 GB | 7.7 GB | no, at any batch size |

The 4-expert model cannot be trained on this card with plain fp32 AdamW. The exact-math workaround
implemented here (`mof/common/cpu_offload_optimizer.py`): master weights, AdamW moments and the
update live on the CPU (same algorithm as `torch.optim.AdamW`, verified bitwise-equal on CPU),
gradients are streamed to pinned CPU buffers during backward, the EMA copy lives on the CPU
(`training.ema_device=cpu`). Cost: ~1 s of CPU/transfer work per optimizer step.

## Run 1 — single-frame "Right" policy on MoveTwoPlates (paper Table 1 row)
- Command (exact paper config; the only extras are "no in-training rollouts" and keeping the
  epoch 460–500 checkpoints for offline evaluation, which is the paper's protocol anyway):
  ```
  ./run.sh train.py --config-name=train_mof_single_frame task=bigym_rby1_move_two_plates \
      'policy.enabled_experts=[right]' policy.canonical_space=right \
      _target_=mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace \
      ++training.enable_rollout=false ++checkpoint.save_epoch_ckpts_from=460 \
      logging.mode=offline hydra.run.dir=data/outputs/sf_right_move_two_plates_seed0
  ```
- 100 demos, batch 128, 500 epochs, AdamW 1e-4, cosine + 500 warm-up, EMA, seed 0. 102 iterations
  per epoch at 1.74 it/s → ~1 min/epoch, 500 epochs ≈ 8.5 h (started 2026-09-10 00:03).
- Evaluation: `notes/scripts/eval_ckpts.py` on epochs 460, 470, 480, 490 and the final epoch 499
  (epochs are 0-indexed in this code, so the paper's "epoch 500" is `latest.ckpt`) × 50 episodes
  (seeds 100000–100049, max 400 steps, score = BiGym task success), 8 EGL envs. Paper uses 3 seeds;
  this run is 1 seed.
- **Paper number:** Right frame on Move 2 Plates = 47.2 ± 1.7 % (MoF-MoE 51.6, DP 38.5).
- Sanity check at epoch 70 (3 CPU episodes): 0/3, but the behaviour is already task-directed (reaches
  the rack, grasps and lifts a plate, drops it) — see `early_eval/contact_sheet.png`.
- **Result (seed 0, 50 episodes per checkpoint):**

  | checkpoint | ep 460 | ep 470 | ep 480 | ep 490 | ep 499 (final) | **mean** |
  |---|---|---|---|---|---|---|
  | success | 44 % | 38 % | 48 % | 30 % | 34 % | **38.8 %** |

  Paper (same policy, 3 seeds): 47.2 ± 1.7 %. One seed here, ~8 points lower; the checkpoint-to-
  checkpoint spread (30–48 %) is the usual DP variance on this task. Training took 7.9 h
  (00:03 → 07:54), evaluation 3.5 min per checkpoint with 8 envs. Raw logs:
  `offline_eval/epoch=*/eval_log.json`, `offline_eval_latest/latest/eval_log.json`.

## Run 2 — MoF-MoE on MoveTwoPlates with the CPU-offload optimizer
- Launched automatically after Run 1's evaluation by `notes/scripts/chain_after_baseline.sh`:
  `train_mof_moe` config, batch 64 × gradient_accumulate_every 2 (= paper batch 128; loss is a
  batch mean and GroupNorm is per-sample, so the gradient is identical up to fp rounding), EMA
  updated once per optimizer step (as in the paper), `CPUOffloadAdamW`, EMA on CPU.
- Deviations from the paper's exact run: none in the math; wall-clock ~30 h instead of ~10 h on a
  large GPU; first optimizer step uses a single micro-batch (artifact of the original
  accumulation loop); 1 seed.
- **Paper number:** MoF-MoE on Move 2 Plates = 51.6 ± 4.4 %.
- **Result:** _pending_

## Videos
- `data/outputs/<run>/videos/*.mp4`: third-person camera + the three policy cameras, one file per
  held-out episode, outcome in the file name. Produced by `notes/scripts/rollout_video.py`.
- `data/outputs/<run>/offline_eval/epoch=XXXX/media/*.mp4`: the runner's own head-camera
  recordings for every evaluated episode.
