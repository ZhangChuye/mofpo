"""Vanilla DexMimicGen dataset for the original DiffPo abs baseline.

Mirrors BigymRBY1ReplayDataset (world-frame obs + abs world-frame action,
plain range normalizer, no frame precomputation). Reuses the HDF5→replay
converter from `dexmimicgen_replay_dataset_mof` so we get the same
20D / 30D action layout and obs key naming.
"""

from typing import Dict, List
import copy
import os
import shutil

import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits

from mof.dataset.base_dataset import BaseImageDataset
from mof.common.pytorch_util import dict_apply
from mof.common.replay_buffer import ReplayBuffer
from mof.common.sampler import SequenceSampler, get_val_mask
from mof.common.normalize_util import (
    array_to_stats,
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_image_identity_normalizer,
    robosuite_dual_arm_action_normalizer_from_stat,
    robosuite_dual_arm_action_normalizer_dex_from_stat,
)
from mof.model.common.normalizer import LinearNormalizer
from mof.dataset.dexmimicgen_replay_dataset_mof import (
    _convert_dexmimicgen_hdf5_to_replay,
)


class DexMimicGenReplayDataset(BaseImageDataset):
    """World-frame obs + abs world-frame action, range normalizer.

    Parallel-jaw tasks: action_dim=20, gripper in [-1, 1] (identity normalize).
    Dex-hand tasks:     action_dim=30, gripper qpos (range normalize).

    `base_pos` / `base_quat` are loaded as constants (no mobile base in
    dexmimicgen) for shape_meta compatibility but are filtered by the policy
    as `_recon_only_obs_keys`, so we skip them in the normalizer.
    """

    def __init__(
        self,
        shape_meta: dict,
        demos_dir: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        n_demo=None,
        normalize_rgb=True,
    ):
        self.normalize_rgb = normalize_rgb
        if not os.path.isabs(demos_dir):
            try:
                from hydra.utils import to_absolute_path as hydra_to_absolute_path
                demos_dir = hydra_to_absolute_path(demos_dir)
            except Exception:
                demos_dir = os.path.abspath(demos_dir)

        dataset_path = demos_dir + ".hdf5"
        if not os.path.exists(dataset_path):
            if os.path.exists(demos_dir) and demos_dir.endswith(".hdf5"):
                dataset_path = demos_dir
            else:
                raise FileNotFoundError(f"Cannot find HDF5 file. Tried: {dataset_path}")

        self.n_demo = n_demo

        cache_dir = os.path.dirname(dataset_path)
        cache_base = os.path.splitext(os.path.basename(dataset_path))[0]

        replay_buffer = None
        if use_cache:
            cache_zarr_path = os.path.join(
                cache_dir, f"cache_dexmimicgen_{cache_base}_{n_demo}.zarr.zip"
            )
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_dexmimicgen_hdf5_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            n_demo=n_demo,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_dexmimicgen_hdf5_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                n_demo=n_demo,
            )

        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        key_first_k: Dict[str, int] = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        action_dim = self.replay_buffer["action"].shape[-1]
        grip_dim = (action_dim - 18) // 2
        action_normalizer_fn = (
            robosuite_dual_arm_action_normalizer_from_stat
            if grip_dim == 1
            else robosuite_dual_arm_action_normalizer_dex_from_stat
        )
        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = action_normalizer_fn(action_stat)

        for key in self.lowdim_keys:
            if key not in self.replay_buffer:
                continue
            stat = array_to_stats(self.replay_buffer[key])
            # `base_pos` / `base_quat` are constants in dexmimicgen (no mobile
            # base): identity-normalize so policies that don't filter them as
            # recon-only (e.g. moe-dp official in `train_action_key='action'`
            # mode) still find a normalizer entry.
            if key in ("base_pos", "base_quat") or key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                this_normalizer = get_range_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = (
                get_image_range_normalizer() if self.normalize_rgb
                else get_image_identity_normalizer()
            )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict: Dict[str, np.ndarray] = {}
        for key in self.rgb_keys:
            # Replay buffer stores RGB as (T, H, W, C) uint8; convert to
            # (T, C, H, W) float32 in [0, 1].
            arr = data[key][T_slice]
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
            if arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = np.moveaxis(arr, -1, 1)
            obs_dict[key] = arr
            del data[key]
        for key in self.lowdim_keys:
            if key not in data:
                continue
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
