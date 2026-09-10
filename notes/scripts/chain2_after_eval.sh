#!/usr/bin/env bash
# Sequel to chain_after_baseline.sh: wait for the 4-checkpoint eval, evaluate the final-epoch
# checkpoint (latest.ckpt = epoch 499, the paper's "epoch 500"), record third-person videos, then
# launch MoF-MoE with the CPU-offload optimizer. Steps run strictly sequentially (RAM-bound).
set -uo pipefail
cd /home/chuye/Documents/mofpo
source /home/chuye/anaconda3/etc/profile.d/conda.sh && conda activate mof
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
BASE=data/outputs/sf_right_move_two_plates_seed0
echo "[chain2] $(date) waiting for $BASE/offline_eval/summary.json"
until [ -f $BASE/offline_eval/summary.json ]; do sleep 20; done
while pgrep -f "^python notes/scripts/eval_ckpts.py" > /dev/null; do sleep 5; done
echo "[chain2] $(date) 4-ckpt eval done: $(tr -d '\n ' < $BASE/offline_eval/summary.json)"

echo "[chain2] $(date) evaluating latest.ckpt (epoch 499) with 50 episodes, 8 envs"
python notes/scripts/eval_ckpts.py --run_dir $BASE --ckpts checkpoints/latest.ckpt --out_dir $BASE/offline_eval_latest --n_envs 8 2>&1 | grep -E "^(===|latest|SUMMARY|Traceback|.*Error)"
echo "[chain2] $(date) latest eval done"

echo "[chain2] $(date) third-person videos from latest.ckpt"
python notes/scripts/rollout_video.py --ckpt $BASE/checkpoints/latest.ckpt --out_dir $BASE/videos --seeds 100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 2>&1 | grep -E "^(seed|success|Traceback|.*Error)"
echo "[chain2] $(date) videos done"

RUN=data/outputs/mof_moe_move_two_plates_seed0_offload
mkdir -p $RUN
echo "[chain2] $(date) launching MoF-MoE (CPU-offload AdamW, batch 64 x accum 2, EMA on CPU) -> $RUN"
nohup ./run.sh train.py --config-name=train_mof_moe task=bigym_rby1_move_two_plates \
  _target_=mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace \
  ++training.enable_rollout=false ++checkpoint.save_epoch_ckpts_from=460 \
  optimizer._target_=mof.common.cpu_offload_optimizer.CPUOffloadAdamW ++optimizer.num_threads=16 ++training.ema_device=cpu \
  dataloader.batch_size=64 training.gradient_accumulate_every=2 \
  logging.mode=offline hydra.run.dir=$RUN > $RUN/train_stdout.log 2>&1 &
echo "[chain2] $(date) MoF-MoE pid $!"
