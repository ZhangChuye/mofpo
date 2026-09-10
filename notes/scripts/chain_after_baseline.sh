#!/usr/bin/env bash
# Morning chain: wait for the baseline training process to exit, evaluate its last-5
# checkpoints (paper protocol) on the GPU, record third-person rollout videos, then start
# the MoF-MoE run with the CPU-offload optimizer (the only exact path on this 8 GB GPU).
set -uo pipefail
cd /home/chuye/Documents/mofpo
source /home/chuye/anaconda3/etc/profile.d/conda.sh && conda activate mof
BASE=data/outputs/sf_right_move_two_plates_seed0
TRAIN_PID=${1:?usage: chain_after_baseline.sh <training_pid>}
echo "[chain] $(date) waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[chain] $(date) training process exited"
ls -la $BASE/checkpoints/
N_EPOCH_CKPTS=$(ls $BASE/checkpoints/epoch=*.ckpt 2>/dev/null | wc -l)
if [ "$N_EPOCH_CKPTS" -lt 1 ]; then echo "[chain] no epoch checkpoints found, aborting"; exit 1; fi

echo "[chain] $(date) offline eval of $N_EPOCH_CKPTS checkpoints (50 episodes each, 8 envs)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python notes/scripts/eval_ckpts.py --run_dir $BASE --n_envs 8 --out_dir $BASE/offline_eval 2>&1 | grep -E "^(===|epoch=|SUMMARY|Traceback|.*Error|\s+\")"
echo "[chain] $(date) eval done"

LAST_CKPT=$(ls $BASE/checkpoints/epoch=*.ckpt | tail -1)
echo "[chain] $(date) third-person videos from $LAST_CKPT"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
  python notes/scripts/rollout_video.py --ckpt "$LAST_CKPT" --out_dir $BASE/videos --seeds 100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 2>&1 | grep -E "^(seed|success|Traceback|.*Error)"
echo "[chain] $(date) videos done"

RUN=data/outputs/mof_moe_move_two_plates_seed0_offload
mkdir -p $RUN
echo "[chain] $(date) launching MoF-MoE (CPU-offload AdamW, batch 64 x accum 2 = 128, EMA on CPU) -> $RUN"
nohup ./run.sh train.py --config-name=train_mof_moe task=bigym_rby1_move_two_plates \
  _target_=mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace \
  ++training.enable_rollout=false ++checkpoint.save_epoch_ckpts_from=460 \
  optimizer._target_=mof.common.cpu_offload_optimizer.CPUOffloadAdamW ++optimizer.num_threads=16 ++training.ema_device=cpu \
  dataloader.batch_size=64 training.gradient_accumulate_every=2 \
  logging.mode=offline hydra.run.dir=$RUN > $RUN/train_stdout.log 2>&1 &
echo "[chain] $(date) MoF-MoE pid $!"
