#!/usr/bin/env bash
# After the MoF-MoE run finishes: evaluate its epoch checkpoints + latest (epoch 499) with the paper
# protocol (50 episodes each), then record third-person videos. Sequential on purpose (RAM-bound:
# ~1.9 GB per env worker). Safe to re-run: pass no PID to start immediately.
#   ./notes/scripts/chain3_after_mof_moe.sh [<training_pid>]
set -uo pipefail
cd /home/chuye/Documents/mofpo
source /home/chuye/anaconda3/etc/profile.d/conda.sh && conda activate mof
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
RUN=data/outputs/mof_moe_move_two_plates_seed0_offload
PID=${1:-}
if [ -n "$PID" ]; then
  echo "[chain3] $(date) waiting for MoF-MoE pid $PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 120; done
  echo "[chain3] $(date) MoF-MoE training exited"
fi
ls -la $RUN/checkpoints/

# The host's NVIDIA userspace libs were upgraded under a running kernel module (2026-09-11), so new
# CUDA processes fail with error 804 until the machine is rebooted. Check before burning hours.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  cat <<'MSG'
[chain3] ACTION REQUIRED: this machine cannot start new CUDA processes (driver/library mismatch,
         CUDA error 804). Evaluation is skipped. Fix with a reboot (or `sudo modprobe -r nvidia_uvm
         nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia`), then re-run:
             ./notes/scripts/chain3_after_mof_moe.sh
         To evaluate without a GPU instead (slow, ~1 h per checkpoint), add --device cpu to the
         eval_ckpts.py calls in this script.
MSG
  exit 2
fi

if [ "$(ls $RUN/checkpoints/epoch=*.ckpt 2>/dev/null | wc -l)" -lt 1 ]; then echo "[chain3] no epoch checkpoints, aborting"; exit 1; fi
echo "[chain3] $(date) eval epoch checkpoints (50 episodes each, 8 envs)"
python notes/scripts/eval_ckpts.py --run_dir $RUN --n_envs 8 --out_dir $RUN/offline_eval 2>&1 | grep -E "^(===|epoch=|SUMMARY|Traceback|.*Error|\s+\")"
echo "[chain3] $(date) eval latest.ckpt (epoch 499)"
python notes/scripts/eval_ckpts.py --run_dir $RUN --ckpts checkpoints/latest.ckpt --n_envs 8 --out_dir $RUN/offline_eval_latest 2>&1 | grep -E "^(===|latest|SUMMARY|Traceback|.*Error)"
echo "[chain3] $(date) videos"
python notes/scripts/rollout_video.py --ckpt $RUN/checkpoints/latest.ckpt --out_dir $RUN/videos --seeds 100000 100001 100002 100003 100004 100005 100006 100007 100008 100009 2>&1 | grep -E "^(seed|success|Traceback|.*Error)"
echo "[chain3] $(date) all done"
