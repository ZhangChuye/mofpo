# MoF (Mixture of Frames Policy) — BiGym RBY1 simulation setup, in brief

Sources: paper arXiv:2607.11884 (Sec. 3.1, 4, App. A–C) and this code base (`mof/config/*`, `mof/env/bigym/factory.py`, BiGym fork `pointW/bigym@2658314`). Every item below was checked against both.

## Robot / simulator / tasks
- **Robot:** RBY1 bimanual *wheeled mobile* manipulator (Robotiq 2F-85 grippers), BiGym benchmark, MuJoCo 3.3.5 (pinned). Base is not actuated directly: a whole-body IK controller moves base + torso + arms to track the commanded end-effector poses.
- **5 tasks:** FlipCup, MoveTwoPlates, StoreKitchenware, FlipSandwich, DishwasherLoadPlates.

| task | control freq | max eval steps | demo length (steps) |
|---|---|---|---|
| FlipCup | 50 Hz | 400 | – (not downloaded yet) |
| MoveTwoPlates | 20 Hz | 400 | 120–170, mean 139 (100 demos) |
| StoreKitchenware | 20 Hz | 800 | – |
| FlipSandwich | 20 Hz | 800 | – |
| DishwasherLoadPlates | 50 Hz | 1300 | – |

## Perception (policy input)
- **RGB only.** No depth, no point cloud (dataset ships RGB + low-dim only).
- **3 cameras: head (egocentric, on the robot head) + LEFT WRIST + RIGHT WRIST.** Yes, two wrist cameras. All three are MuJoCo cameras with fovy 90°, mounted on the robot (wrist cams on each gripper body).
- Resolution: 84×84 (FlipCup is stored/rendered at 224×224 in the code; the paper text says 84×84 for all). Encoder pipeline for every task: ImageNet-normalize → resize 224 → random crop 192 (train) / center crop (test) → resize 76×76.
- Image history: 1 frame (`img_obs_horizon=1`). Low-dim history: 2 steps (`n_obs_steps=2`).
- **Low-dim obs (per step):** gripper state (2), left EE pos (3) + quat (4), right EE pos (3) + quat (4), head-site pos (3) + quat (4). Base pos/quat (world) are also read but used *only* to build the frame transforms, not fed to the encoder. Proprio is re-expressed in each expert's frame before conditioning that expert.

## Policy (Diffusion Policy backbone + MoF)
- Backbone: standard Diffusion Policy. ResNet-18 (ImageNet-pretrained, fine-tuned, spatial-softmax, GroupNorm, **separate encoder per camera**), conditional 1D U-Net [256, 512, 1024], FiLM conditioning, step-embed 128, kernel 5, 8 groups. DDIM, 50 train steps / 16 inference steps, ε-prediction. EMA weights used at test time.
- **Output:** action chunk of 16 steps, first 8 executed, then re-plan.
- **MoF-MoE:** 4 U-Net "experts", one per frame — `base_rel_trans` (base-relative rotation/translation direction, translation re-centred at each arm's current EE position), `left` (left EE frame), `right` (right EE frame), `rel_traj` (each arm's motion in its own current EE frame). Canonical frame = `base_rel_trans`. At every denoising step the same canonical noisy action is re-expressed in each expert's frame (6-D rotation stored as matrix *columns* so the transform is linear), every expert predicts noise, predictions are mapped back and fused with a learned softmax router (MLP, hidden 512; input = shared RGB features + per-frame low-dim + diffusion-step embedding). Aux per-expert loss weight 1.0. **MoF-Ensemble** = same, uniform weights 1/4.

## Action (policy output) — 20-D, absolute, world frame
`[left EE pos (3), left rot-6D (6), right EE pos (3), right rot-6D (6), left gripper (1), right gripper (1)]`; gripper commands are binary 0/1; poses are absolute world-frame targets tracked by the whole-body IK. No base action.

## Training data
- **100 demos per task** (the release has 200 files for FlipCup and MoveTwoPlates; configs use the first 100).
- Generated automatically: the authors implemented the **DexMimicGen** data-generation algorithm and extended it to mobile manipulation (paper App. C.1). The BiGym fork's scripts show the source demos were BiGym's human demos converted from the H1 robot to RBY1 Cartesian actions (this last detail is from the fork's scripts, not stated in the paper).
- Hosted at HF `dian-wang/mof-datasets`, one `.safetensors` per demo (RGB ×3, low-dim, 20-D action, success flags). Sizes: FlipCup 32.6 GB (224 px), MoveTwoPlates 1.8 GB, StoreKitchenware 4.0 GB, FlipSandwich 2.4 GB, DishwasherLoadPlates 6.7 GB.

## Training / evaluation protocol
- Batch 128, 500 epochs, AdamW lr 1e-4, wd 1e-6, betas (0.95, 0.999), cosine schedule with 500 warm-up steps, EMA; 2 % of demos held out for val loss. 3 seeds.
- Eval: for each seed, checkpoints at epochs 460, 470, 480, 490, 500 × 50 rollouts each (seeds 100000+, disjoint from training), 25 parallel envs; score = BiGym sparse task success. Reported number = mean over 5 ckpts × 3 seeds, ± std. err. over seeds.

## Paper numbers to match (success %, Table 2)
| | FlipCup | MoveTwoPlates | Kitchenware | FlipSandwich | Dishwasher |
|---|---|---|---|---|---|
| MoF-MoE | 56.5 ± 1.4 | 51.6 ± 4.4 | 31.6 ± 2.4 | 40.0 ± 3.1 | 89.2 ± 3.9 |
| MoF-Ensemble | 59.2 ± 0.4 | 51.5 ± 1.5 | 33.7 ± 0.5 | 43.2 ± 2.0 | 89.7 ± 1.4 |
| DP (world frame) | 17.2 ± 2.8 | 38.5 ± 0.4 | 22.0 ± 1.6 | 32.5 ± 3.1 | 61.5 ± 2.1 |

## Checkpoints
**None released** (no GitHub releases, no HF model repo, README has no checkpoint link). Reproduction has to train from scratch.
