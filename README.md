# Mixture of Frames Policy (MoF)

Official code for **"Mixture of Frames Policy: Multi-Frame Action Denoising for
Bimanual Mobile Manipulation"**.

<img width="100%" src="https://mofpo.github.io/images/teaser.jpg">

Mixture of Frames (MoF) is a diffusion policy that denoises each action in
**several reference frames at once** — a base-relative frame, the left and right
end-effector frames, and a trajectory-relative frame — and fuses the per-frame
predictions either with a learned per-timestep router (**MoF-MoE**) or with a
uniform average (**MoF-Ensemble**). Denoising in multiple frames exposes the
policy to several complementary action parameterizations of the same motion,
which improves bimanual mobile-manipulation performance over any single frame.

This repository contains the **simulation** experiments from the paper, on the
**BiGym** (RBY1) and **DexMimicGen** task suites. The real-world experiments
(Appendix F, a DiT + HoMMI stack) live in the
[`mof_hommi`](https://github.com/gsanpark/mof_hommi) submodule — clone with
`git clone --recurse-submodules`, or `git submodule update --init --recursive mof_hommi` in
an existing checkout.

---

## Installation

Clone the repository and `cd` into it, then:

```bash
# 1. System packages for MuJoCo
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# 2. Create and activate the conda environment (all version pins live here)
mamba env create -f conda_environment.yaml   # or: conda env create -f ...
conda activate mof

# 3. Install the MoF stack into the active env
bash install.sh
```

`install.sh` installs this package plus the pinned robosuite / robomimic /
robosuite-task-zoo / DexMimicGen / BiGym sources into the **active** conda env
(it aborts if `mof` isn't active, so it won't pollute another env). Almost
everything is pinned in `conda_environment.yaml`; the one exception baked into
the script is `mink==1.1.0`, installed `--no-deps` because it requires
`mujoco>=3.3.6` while this project deliberately holds `mujoco==3.3.5`.

---

## Datasets

The nine demonstration sets are on the HuggingFace Hub
([`dian-wang/mof-datasets`](https://huggingface.co/datasets/dian-wang/mof-datasets))
— RGB + low-dim observations only (the depth / point-cloud channels are dropped
since the released image policies never read them). Use the helper to fetch them
straight into `data/`:

```bash
python -m mof.scripts.download_datasets                 # all 9 tasks
python -m mof.scripts.download_datasets --tasks bigym   # all 5 BiGym  (or: --tasks dex)
python -m mof.scripts.download_datasets --tasks rby1_flip_cup two_arm_threading
python -m mof.scripts.download_datasets --list          # list tasks, paths, and task= names
```

It downloads to exactly the layout the task configs expect (each `task=` config
points at its own `demos_dir`, so nothing else needs setting) — after download you
can train directly with `task=<task>`:

```
data/bigym/rby1_flip_cup/*.safetensors            # FlipCup (224x224)
data/bigym/rby1_move_two_plates/*.safetensors     # the other four are native 84x84
data/bigym/rby1_store_kitchenware/*.safetensors
data/bigym/rby1_flip_sandwich/*.safetensors
data/bigym/rby1_dishwasher_load_plates/*.safetensors
data/dexmimicgen/two_arm_threading_abs.hdf5       # + three_piece_assembly / box_cleanup / drawer_cleanup
```

The DexMimicGen sets are already absolute-action (`_abs`). To regenerate them from
raw delta-action demos instead, use
[`mof/scripts/dexmimicgen_dataset_conversion.py`](mof/scripts/dexmimicgen_dataset_conversion.py):

```bash
python -m mof.scripts.dexmimicgen_dataset_conversion \
    -i data/dexmimicgen/two_arm_threading.hdf5 \
    -o data/dexmimicgen/two_arm_threading_abs.hdf5 \
    -e data/dexmimicgen/two_arm_threading_eval -n 8
```

---

## Reproducing the paper

Every paper run launches through [`run_async_eval.sh`](run_async_eval.sh), which
runs training while periodically rolling out the policy for evaluation in a
separate GPU context:

```bash
CUDA_VISIBLE_DEVICES=<gpu> ./run_async_eval.sh train.py \
    --config-name=<config> task=<task> [overrides...]
```

Settings shared by all paper runs are **baked into the configs** (no override
needed): `n_demo=100`, `dataloader.batch_size=128`, a ResNet18 (timm
`resnet18.tv_in1k`, spatial-softmax) vision encoder at a `84→224→crop192→76`
input, a DDIM scheduler (50 train / 16 inference steps), and 500 epochs.

### MoF policy knobs

The MoF method is fully described by four `policy.*` fields (all baked into
`train_mof_moe` for MoF-MoE):

| Field | Values | Meaning |
|---|---|---|
| `enabled_experts` | any subset of `[base_rel_trans, left, right, rel_traj]` | which frame experts to denoise in |
| `canonical_space` | one of the enabled frames | the frame the shared canonical action (diffusion target + noise `x_T`) is expressed in; every expert's prediction is mapped into it before fusing |
| `router_mode` | `learned` \| `fixed_uniform` | learned per-timestep router (MoF-MoE) vs. uniform average (MoF-Ensemble) |
| `expert_loss_coef` | float (`1.0` default, `0.0` = off) | weight of the per-expert auxiliary loss |

The four frames are: `base_rel_trans` (base-relative translation, pseudo-base
from the head pose), `left` / `right` (end-effector frames), and `rel_traj`
(trajectory-relative).

### Tasks

The paper evaluates on **nine** simulation tasks. Each is a **self-contained
`task=` config** — `shape_meta`, env, `task_name`, and `demos_dir` are all baked
in, so selecting a task is just one flag and nothing else. Task selection is
orthogonal to the method config.

**BiGym (RBY1)** — `task=bigym_rby1_<name>`:
`flip_cup`, `move_two_plates`, `store_kitchenware`, `flip_sandwich`,
`dishwasher_load_plates`. (The observation encoder maps every task to the same
input size, so the policy sees a uniform resolution. The only difference is the
*source* image resolution baked into each task config — FlipCup's frames are
224×224, the other four 84×84 — which the encoder resizes; FlipCup keeps its
higher source resolution because downsampling it measurably hurts that task.)

**DexMimicGen** — `task=dexmimicgen_<name>`:
`two_arm_threading`, `two_arm_three_piece_assembly` (parallel-jaw, 20-D action),
`two_arm_box_cleanup`, `two_arm_drawer_cleanup` (dex hand, 30-D action).

The dataset *class* is chosen by the **method** config (the MoF configs use the
MoF dataset); the task config only fixes the family (BiGym vs DexMimicGen)
and the demos path. Other envs exist in
[`mof/env/bigym/factory.py`](mof/env/bigym/factory.py) and the `get_max_steps`
table in [`train.py`](train.py); the nine above are the paper set.

### Example launches

Every launch is `--config-name=<method> task=<task>` plus any method overrides.
The task config carries `task_name`, `demos_dir`, `shape_meta` and the env.

```bash
# MoF-MoE on a BiGym task (FlipCup is 224 — handled by the task config)
CUDA_VISIBLE_DEVICES=0 ./run_async_eval.sh train.py \
    --config-name=train_mof_moe task=bigym_rby1_flip_cup

# MoF-Ensemble on a BiGym task
CUDA_VISIBLE_DEVICES=0 ./run_async_eval.sh train.py \
    --config-name=train_mof_ensemble task=bigym_rby1_store_kitchenware

# Single-frame (Left frame) baseline
CUDA_VISIBLE_DEVICES=0 ./run_async_eval.sh train.py \
    --config-name=train_mof_single_frame task=bigym_rby1_flip_cup \
    policy.enabled_experts=[left] policy.canonical_space=left

# MoF-MoE on a DexMimicGen task (parallel-jaw: threading/assembly;
# dex-hand: box_cleanup/drawer_cleanup)
CUDA_VISIBLE_DEVICES=0 ./run_async_eval.sh train.py \
    --config-name=train_mof_moe task=dexmimicgen_two_arm_threading
```

To train without async evaluation (no rollouts), launch through
[`run.sh`](run.sh) instead of `run_async_eval.sh`.

---

## Repository layout

```
mof/
  config/        Hydra configs — train_*.yaml (methods) + task/ (task groups)
  policy/        MixtureOfFramesPolicy (+ GramSchmidt), DP / Ensemble / MoE-DP baselines
  model/         diffusion UNet (+ MoF / ensemble variants), MoE module, vision encoder
  dataset/       per-frame replay datasets (BiGym RBY1, DexMimicGen)
  env/           BiGym + DexMimicGen environment wrappers and factories
  env_runner/    async rollout / evaluation runners
  workspace/     training loop, EMA, checkpointing, async-eval workspace
  scripts/       dataset conversion
train.py         training entrypoint     eval.py   evaluation entrypoint
run.sh           single-GPU launch       run_async_eval.sh  launch with async eval
```

## License

Released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgement

Built upon [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
The MoE-DP baseline is a faithful port of the official
[MoE-DP](https://github.com/AlanxChen/moedp) release. Tasks come from
[BiGym](https://github.com/chernyadev/bigym) and
[DexMimicGen](https://github.com/NVlabs/dexmimicgen).
