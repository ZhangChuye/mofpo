#!/usr/bin/env bash
# After the MoF-MoE offload run exits: evaluate its checkpoints 460/470/480/490 + latest (epoch 499)
# with the paper protocol (50 episodes each), then record third-person videos. Sequential (RAM-bound).
set -uo pipefail
cd /home/chuye/Documents/mofpo
source /home/chuye/anaconda3/etc/profile.d/conda.sh && conda activate mof
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
RUN=data/outputs/mof_moe_move_two_plates_seed0_offload
PID=${1:?usage: chain3_after_mof_moe.sh <training_pid>}
echo "[chain3] $(date) waiting for MoF-MoE pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "[chain3] $(date) MoF-MoE training exited; checkpoints:"; ls -la $RUN/checkpoints/
if [ "$(ls $RUN/checkpoints/epoch=*.ckpt 2>/dev/null | wc -l)" -lt 1 ]; then echo "[chain3] no epoch checkpoints, aborting"; exit 1; fi
echo "[chain3] $(date) eval epoch checkpoints (50 episodes each, 8 envs)"
python notes/scripts/eval_ckpts.py --run_dir $RUN --n_envs 8 --out_dir $RUN/offline_eval 2>&1 | grep -E "^(===|epoch=|SUMMARY|Traceback|.*Error|\s+\")"
echo "[chain3] $(date) eval latest.ckpt (epoch 499)"
python notes/scripts/eval_ckpts.py --run_dir $RUN --ckpts checkpoints/latest.ckpt --n_envs 8 --out_dir $RUN/offline_eval_latest 2>&1 | grep -E "^(===|latest|SUMMARY|Traceback|.*Error)"
echo "[chain3] $(date) videos"
python notes/scripts/rollout_video.py --ckpt $RUN/checkpoints/latest.ckpt --out_dir $RUN/videos --seeds 100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 2>&1 | grep -E "^(seed|success|Traceback|.*Error)"
echo "[chain3] $(date) all done"
