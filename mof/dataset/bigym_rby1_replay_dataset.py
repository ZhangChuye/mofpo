from typing import Dict, List
import os
import shutil
import copy
import numpy as np
import torch
import zarr
from safetensors.numpy import load_file
from filelock import FileLock
from threadpoolctl import threadpool_limits
import tqdm

from mof.dataset.base_dataset import BaseImageDataset
from mof.common.pytorch_util import dict_apply
from mof.model.common.normalizer import LinearNormalizer
from mof.common.replay_buffer import ReplayBuffer
from mof.common.sampler import SequenceSampler, get_val_mask
from mof.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_image_identity_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    bigym_action_only_normalizer_from_stat,
)
from mof.model.common.rotation_transformer import RotationTransformer
from mof.codecs.imagecodecs_numcodecs import register_codecs

register_codecs(verbose=False)


class BigymRBY1ReplayDataset(BaseImageDataset):
    """
    Dataset variant that ingests the raw .safetensors demos produced for the
    RBY1 ReachTarget task without converting them to npz first.
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

        self.n_demo = n_demo

        replay_buffer = None
        if use_cache:
            # Match the cache filename convention used by mof / rel_traj
            # datasets so the same cache_rby1_<n_demo>.zarr.zip in GCS works
            # across all bigym_rby1_* dataset classes.
            cache_zarr_path = os.path.join(demos_dir, f"cache_rby1_{n_demo}.zarr.zip")
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_rby1_safetensors_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            demos_dir=demos_dir,
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
            replay_buffer = _convert_rby1_safetensors_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                demos_dir=demos_dir,
                n_demo=n_demo,
            )

        rgb_keys: List[str] = list()
        lowdim_keys: List[str] = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
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
        # Plain per-dimension range normalization, matching corl_exp's mof
        # pipeline (bigym_action_only_normalizer_from_stat). No shared XY scale,
        # no symmetric-zero offset.
        normalizer = LinearNormalizer()

        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = bigym_action_only_normalizer_from_stat(action_stat)

        for key in self.lowdim_keys:
            # base_pos / base_quat are in shape_meta but the safetensors
            # converter doesn't load them; the policy filters them as
            # `_recon_only_obs_keys` and never normalizes them.
            if key not in self.replay_buffer:
                continue
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("quat"):
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

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            # Some shape_meta keys (e.g. base_pos / base_quat) aren't loaded
            # from safetensors; the policy filters them as recon-only inputs
            # so we don't need to feed them to the encoder.
            if key not in data:
                continue
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_rby1_safetensors_to_replay(store, shape_meta, demos_dir, n_demo=None):
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    demos_dir_abs = os.path.abspath(demos_dir)

    try:
        dir_entries = os.listdir(demos_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Demo directory '{demos_dir}' does not exist. "
            "Set task.dataset.demos_dir to a folder containing .safetensors demos."
        ) from exc

    files = sorted(
        [
            os.path.join(demos_dir, f)
            for f in dir_entries
            if f.endswith(".safetensors")
        ]
    )
    if not files:
        preview_count = 10
        entries_preview = ", ".join(sorted(dir_entries)[:preview_count])
        if len(dir_entries) > preview_count:
            entries_preview += ", ..."

        sibling_hint = ""
        parent_dir = os.path.dirname(demos_dir_abs)
        sibling_dirs = []
        if os.path.isdir(parent_dir):
            for name in sorted(os.listdir(parent_dir)):
                sibling_path = os.path.join(parent_dir, name)
                if not os.path.isdir(sibling_path):
                    continue
                try:
                    if os.path.samefile(sibling_path, demos_dir_abs):
                        continue
                except OSError:
                    continue
                try:
                    has_safetensors = any(
                        entry.endswith(".safetensors")
                        for entry in os.listdir(sibling_path)
                    )
                except OSError:
                    continue
                if has_safetensors:
                    sibling_dirs.append(name)
                if len(sibling_dirs) >= 5:
                    break
        if sibling_dirs:
            sibling_hint = (
                " Nearby directories containing .safetensors demos: "
                + ", ".join(sibling_dirs)
                + "."
            )

        raise FileNotFoundError(
            f"No '.safetensors' demos found in '{demos_dir}'. "
            f"Directory entries: [{entries_preview or '(empty directory)'}]."
            f"{sibling_hint}"
        )
    if n_demo is not None:
        files = files[:n_demo]

    episode_ends: List[int] = []
    prev_end = 0

    rgb_mapping = {
        "head_image": "obs_rgb_head",
        "left_wrist_image": "obs_rgb_left_wrist",
        "right_wrist_image": "obs_rgb_right_wrist",
    }
    lowdim_mapping = {
        "proprioception": "obs_proprioception",
        "proprioception_grippers": "obs_proprioception_grippers",
        "left_ee_pos": "obs_left_ee_pos",
        "left_ee_quat": "obs_left_ee_quat",
        "right_ee_pos": "obs_right_ee_pos",
        "right_ee_quat": "obs_right_ee_quat",
        "head_site_quat": "obs_head_site_quat",
        "head_site_pos": "obs_head_site_pos",
    }

    rgb_keys = list(rgb_mapping.keys())
    lowdim_keys = list(lowdim_mapping.keys())

    all_lowdim = {k: [] for k in lowdim_keys}
    all_images = {k: [] for k in rgb_keys}
    all_actions: List[np.ndarray] = []

    for path in files:
        data = load_file(path)
        actions = data["info_demo_action"].astype(np.float32)
        T = actions.shape[0]

        for key, tensor_key in lowdim_mapping.items():
            if tensor_key not in data:
                raise KeyError(f'"{tensor_key}" missing in {path}')
            arr = data[tensor_key].astype(np.float32)
            all_lowdim[key].append(arr)

        for key, tensor_key in rgb_mapping.items():
            if tensor_key not in data:
                raise KeyError(f'"{tensor_key}" missing in {path}')
            img = data[tensor_key]
            if img.ndim != 4:
                raise ValueError(f"Expected (T,C,H,W) for {tensor_key}, got {img.shape}")
            img = np.moveaxis(img, 1, -1)  # (T,C,H,W) -> (T,H,W,C)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            all_images[key].append(img)

        all_actions.append(actions)
        prev_end += T
        episode_ends.append(prev_end)

    _ = meta_group.array(
        "episode_ends",
        episode_ends,
        dtype=np.int64,
        compressor=None,
        overwrite=True,
    )

    for key in lowdim_keys + ["action"]:
        if key == "action":
            this_data = np.concatenate(all_actions, axis=0)
        else:
            this_data = np.concatenate(all_lowdim[key], axis=0)
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

    for key in rgb_keys:
        shape = tuple(shape_meta["obs"][key]["shape"])
        c, h, w = shape
        n_steps = episode_ends[-1] if episode_ends else 0
        img_arr = data_group.require_dataset(
            name=key,
            shape=(n_steps, h, w, c),
            chunks=(1, h, w, c),
            compressor=None,
            dtype=np.uint8,
        )
        start = 0
        for seq in all_images[key]:
            T = seq.shape[0]
            img_arr[start : start + T] = seq
            start += T

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
