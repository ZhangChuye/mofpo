#!/usr/bin/env bash
#
# Installs the MoF Python stack into the *currently active* conda env.
#
# Run this AFTER creating and activating the env:
#     mamba env create -f conda_environment.yaml   # or: conda env create -f ...
#     conda activate mof
#     bash install.sh
#
# It covers the pip layer only (the former README steps 3-5). System packages
# (libosmesa6-dev, ...) and the conda env itself are prerequisites, see README.
# Safe to re-run.
#
# Env vars:
#   SRC_DIR   where the editable git checkouts of dexmimicgen / bigym are placed
#             (default: <repo>/src)
#   MOF_ENV   expected active conda env name (default: mof)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$REPO_DIR/src}"
EXPECT_ENV="${MOF_ENV:-mof}"

# --- Guard: refuse to install into the wrong env -----------------------------
# Prevents silently polluting base / another project's env.
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECT_ENV" ]]; then
  echo "ERROR: conda env '$EXPECT_ENV' is not active (CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}')." >&2
  echo "       Run:  mamba env create -f conda_environment.yaml && conda activate $EXPECT_ENV" >&2
  echo "       (If you renamed the env, pass MOF_ENV=<name>.)" >&2
  exit 1
fi

echo ">>> Installing MoF into conda env '$CONDA_DEFAULT_ENV'"
echo ">>> python:        $(command -v python)"
echo ">>> editable src:  $SRC_DIR"
cd "$REPO_DIR"

# --- 1. This package ---------------------------------------------------------
pip install -e .

# --- 2. robosuite (+deps) and the robomimic / task-zoo / DexMimicGen forks ----
# robosuite's own reqs are loose (mujoco>=3.3.0, scipy>=1.2.3, ...) so they
# accept the env's pins unchanged. The rest use --no-deps to keep the pins.
pip install https://github.com/ARISE-Initiative/robosuite/archive/aaa8b9b214ce8e77e82926d677b4d61d55e577ab.tar.gz
pip install --no-deps https://github.com/ARISE-Initiative/robomimic/archive/9ce065180dbcc38ddf8fa4218daa2e75b5fdc28a.tar.gz
pip install --no-deps https://github.com/BoceHu/robosuite-task-zoo/archive/96e3e5cb06553655be7aa6ced03a98b52fdd2981.tar.gz
pip install --no-deps --src "$SRC_DIR" -e "git+https://github.com/NVlabs/dexmimicgen.git@940e8a1b3ad70eb1925ada6b364b197de6bb2af9#egg=dexmimicgen"

# --- 3. mink + BiGym ---------------------------------------------------------
# Both are installed --no-deps so neither touches the env's pinned versions.
# mink requires mujoco>=3.3.6 but we hold mujoco==3.3.5 (its QP backend daqp is
# already pinned in the env). BiGym is --no-deps too: every runtime dep it needs
# is already provided by this env (pyquaternion is added to conda_environment.yaml
# for exactly this reason; pyyaml comes in via hydra), so we keep BiGym's own
# installer from re-resolving or upgrading anything. -e keeps a working tree so
# bigym.ik imports. BiGym's setup.py is self-contained and version-matched for
# standalone `pip install bigym` users, but here the env stays authoritative.
pip install --no-deps mink==1.1.0
pip install --no-deps --src "$SRC_DIR" -e "git+https://github.com/pointW/bigym.git@265831473ed252201ea7e5ecd8d6cab739b8b4c9#egg=bigym"

echo ">>> MoF install complete in env '$CONDA_DEFAULT_ENV'."
