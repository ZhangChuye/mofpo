#!/usr/bin/env bash
# After the MoF-MoE run finishes: evaluate its epoch checkpoints + latest (epoch 499) with the paper
# protocol (50 episodes each), then record third-person videos. Sequential on purpose (RAM-bound:
# ~1.9 GB per env worker). Safe to re-run: pass no PID to start immediately.
#   ./notes/scripts/chain3_after_mof_moe.sh [<training_pid>]
#
# GPU vs CPU: this host's NVIDIA userspace libraries were upgraded under a running kernel module
# (2026-09-11), so new CUDA processes fail with error 804 until a reboot. EGL rendering still works,
# so the script falls back to running the *policy* on the CPU, which is slower but produces the same
# numbers (the rollouts are deterministic given the seeds; only the arithmetic backend differs).
set -uo pipefail
cd /home/chuye/Documents/mofpo
source /home/chuye/anaconda3/etc/profile.d/conda.sh && conda activate mof
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
RUN=data/outputs/mof_moe_move_two_plates_seed0_offload
PID=${1:-}
if [ -n "$PID" ]; then
  echo "[chain3] $(date) waiting for MoF-MoE pid $PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 120; done
  echo "[chain3] $(date) MoF-MoE training exited"
  sleep 60   # let the last checkpoint finish being written
fi
ls -la $RUN/checkpoints/

if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEV=cuda:0; ENVS=8; export CUDA_VISIBLE_DEVICES=0
  echo "[chain3] $(date) CUDA available -> policy on GPU, $ENVS env workers"
else
  DEV=cpu; ENVS=6; export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  echo "[chain3] $(date) NO CUDA (driver/library mismatch, error 804) -> policy on CPU, $ENVS env workers."
  echo "[chain3]            A reboot restores GPU evaluation; re-run this script afterwards for a faster pass."
fi

if [ "$(ls $RUN/checkpoints/epoch=*.ckpt 2>/dev/null | wc -l)" -lt 1 ]; then echo "[chain3] no epoch checkpoints, aborting"; exit 1; fi
echo "[chain3] $(date) eval epoch checkpoints (50 episodes each, $ENVS envs, device=$DEV)"
python notes/scripts/eval_ckpts.py --run_dir $RUN --n_envs $ENVS --device $DEV --out_dir $RUN/offline_eval 2>&1 | grep -E "^(===|epoch=|SUMMARY|Traceback|.*Error|\s+\")"
echo "[chain3] $(date) eval latest.ckpt (epoch 499)"
python notes/scripts/eval_ckpts.py --run_dir $RUN --ckpts checkpoints/latest.ckpt --n_envs $ENVS --device $DEV --out_dir $RUN/offline_eval_latest 2>&1 | grep -E "^(===|latest|SUMMARY|Traceback|.*Error)"
echo "[chain3] $(date) videos"
python notes/scripts/rollout_video.py --ckpt $RUN/checkpoints/latest.ckpt --device $DEV --out_dir $RUN/videos --seeds 100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 2>&1 | grep -E "^(seed|success|Traceback|.*Error)"
echo "[chain3] $(date) all done"
