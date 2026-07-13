#!/usr/bin/env bash
#
# Single-GPU launch script that overrides the workspace
# to a dedicated async-eval variant.
#
# Usage:
#   ./run_async_eval.sh train.py --config-name=train_mof_moe task=bigym_rby1_flip_cup
#   ASYNC_EVAL_GPU_ID=1 ./run_async_eval.sh train.py --config-name=train_dp task=bigym_rby1_flip_cup
#   ASYNC_WORKSPACE_TARGET=mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace \
#     ./run_async_eval.sh train.py --config-name=... [other args]
#

set -e

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_DEVICES="${CUDA_DEVICES// /}"  # strip spaces
IFS=',' read -ra GPUS <<< "$CUDA_DEVICES"
FIRST_GPU="${GPUS[0]:-0}"
ASYNC_EVAL_GPU_ID="${ASYNC_EVAL_GPU_ID:-$FIRST_GPU}"

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

CONFIG_NAME=""
prev=""
for arg in "$@"; do
    if [[ "$arg" == --config-name=* ]]; then
        CONFIG_NAME="${arg#--config-name=}"
    elif [[ "$prev" == "--config-name" ]]; then
        CONFIG_NAME="$arg"
    fi
    prev="$arg"
done

infer_workspace_target() {
    case "$1" in
        train_*)
            # Every config in this repo (train_mof_*, train_dp, train_moe_dp,
            # train_single_frame_ensemble) uses the same base workspace, so the
            # async-eval variant applies uniformly.
            echo "mof.workspace.train_diffusion_unet_hybrid_async_eval_workspace.TrainDiffusionUnetHybridAsyncEvalWorkspace"
            ;;
        *)
            echo ""
            ;;
    esac
}

WORKSPACE_TARGET="${ASYNC_WORKSPACE_TARGET:-}"
if [[ -z "$WORKSPACE_TARGET" ]]; then
    WORKSPACE_TARGET="$(infer_workspace_target "$CONFIG_NAME")"
fi

if [[ -z "$WORKSPACE_TARGET" ]]; then
    echo "Could not infer async workspace target from --config-name." >&2
    echo "Set ASYNC_WORKSPACE_TARGET explicitly." >&2
    exit 1
fi

exec python "$@" \
    _target_="$WORKSPACE_TARGET" \
    ++training.enable_rollout=true \
    ++training.async_eval_gpu_id="$ASYNC_EVAL_GPU_ID"
