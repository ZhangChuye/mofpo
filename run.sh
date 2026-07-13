#!/usr/bin/env bash
#
# Single-GPU launch script with standard env setup.
#
# Usage:
#   ./run.sh train.py --config-name=train_mof_moe task=bigym_rby1_flip_cup
#   CUDA_VISIBLE_DEVICES=2 ./run.sh train.py --config-name=... [other args]
#

set -e

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_DEVICES="${CUDA_DEVICES// /}"  # strip spaces
IFS=',' read -ra GPUS <<< "$CUDA_DEVICES"
FIRST_GPU="${GPUS[0]:-0}"

export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
export EGL_DEVICE_ID="$FIRST_GPU"
export MUJOCO_EGL_DEVICE_ID="$FIRST_GPU"

# OpenMP env for stability (user can override)
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_WARNINGS="${KMP_WARNINGS:-0}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

# EGL for MuJoCo (user can override)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec python "$@"
