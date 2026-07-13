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
from mof.dataset.bigym_rby1_precompute_util import (
    gather_padded_chunks_from_full_array,
    gather_padded_timestep_from_full_array,
)

register_codecs(verbose=False)


class BigymRBY1ReplayDatasetRelTraj(BaseImageDataset):
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
        if not os.path.isabs(demos_dir):
            try:
                from hydra.utils import to_absolute_path as hydra_to_absolute_path

                demos_dir = hydra_to_absolute_path(demos_dir)
            except Exception:
                demos_dir = os.path.abspath(demos_dir)

        self.n_demo = n_demo
        self.normalize_rgb = normalize_rgb

        replay_buffer = None
        cache_precompute_path = None
        cache_lock_path = None
        if use_cache:
            cache_zarr_path = os.path.join(demos_dir, f"cache_rby1_{n_demo}.zarr.zip")
            cache_precompute_path = cache_zarr_path + ".rel_traj_base_frame.precomputed.npz"
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
            key_first_k["base_pos"] = n_obs_steps
            key_first_k["base_quat"] = n_obs_steps

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
        self.key_first_k = key_first_k
        self.sixd_to_mat = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")
        self.quat_to_mat = RotationTransformer(from_rep="quaternion", to_rep="matrix")
        self.mat_to_quat = RotationTransformer(from_rep="matrix", to_rep="quaternion")
        loaded_precomputed = False
        if use_cache and cache_precompute_path is not None:
            with FileLock(cache_lock_path):
                if os.path.exists(cache_precompute_path):
                    loaded_precomputed = self._load_precomputed(cache_precompute_path)
                    if loaded_precomputed:
                        print("Loaded precomputed rel-action cache.")

        if not loaded_precomputed:
            self.base_frame_lowdim = self._precompute_base_frame_lowdim()
            (
                self.actions,
                self.rel_actions,
            ) = self._precompute_action_chunks()
            if use_cache and cache_precompute_path is not None:
                with FileLock(cache_lock_path):
                    if os.path.exists(cache_precompute_path):
                        loaded_precomputed = self._load_precomputed(cache_precompute_path)
                        if loaded_precomputed:
                            print("Loaded precomputed rel-action cache.")
                    if not loaded_precomputed:
                        self._save_precomputed(cache_precompute_path)

    def _save_precomputed(self, path: str):
        payload = {
            "meta_horizon": np.int64(self.horizon),
            "meta_n_obs_steps": np.int64(-1 if self.n_obs_steps is None else self.n_obs_steps),
            "actions": self.actions,
            "rel_actions": self.rel_actions,
        }
        for key in self.lowdim_keys + ["base_pos", "base_quat"]:
            payload[f"base_lowdim__{key}"] = self.base_frame_lowdim[key]
            payload[f"obs_chunk__{key}"] = self.base_frame_obs_chunks[key]
        np.savez_compressed(path, **payload)
        print(f"Saved precomputed rel-action cache to: {path}")

    def _load_precomputed(self, path: str) -> bool:
        try:
            cache = np.load(path)
        except Exception as exc:
            print(f"Failed to load precomputed cache {path}: {exc}")
            return False

        expected_horizon = int(self.horizon)
        expected_n_obs = -1 if self.n_obs_steps is None else int(self.n_obs_steps)
        got_horizon = int(cache["meta_horizon"])
        got_n_obs = int(cache["meta_n_obs_steps"])
        if got_horizon != expected_horizon or got_n_obs != expected_n_obs:
            print(
                "Precomputed cache shape settings mismatch; recomputing. "
                f"(horizon {got_horizon}!={expected_horizon} or "
                f"n_obs_steps {got_n_obs}!={expected_n_obs})"
            )
            return False

        try:
            self.base_frame_lowdim = {
                key: cache[f"base_lowdim__{key}"] for key in self.lowdim_keys + ["base_pos", "base_quat"]
            }
            self.base_frame_obs_chunks = {
                key: cache[f"obs_chunk__{key}"] for key in self.lowdim_keys + ["base_pos", "base_quat"]
            }
            self.actions = cache["actions"]
            self.rel_actions = cache["rel_actions"]
        except KeyError as exc:
            print(f"Precomputed cache missing key {exc}; recomputing.")
            return False

        return True

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
        val_set._precompute_base_frame_obs_chunks()
        (
            val_set.actions,
            val_set.rel_actions,
        ) = val_set._precompute_action_chunks()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = bigym_action_only_normalizer_from_stat(stat)


        normalizer["rel_action"] = bigym_action_only_normalizer_from_stat(
            array_to_stats(self.rel_actions.reshape(-1, self.rel_actions.shape[-1]))
        )

        for key in self.lowdim_keys:
            stat = array_to_stats(self.base_frame_lowdim[key])
            if key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                this_normalizer = get_range_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        base_pos_stat = array_to_stats(self.base_frame_lowdim["base_pos"])
        normalizer["base_pos"] = get_range_normalizer_from_stat(base_pos_stat)
        base_quat_stat = array_to_stats(self.base_frame_lowdim["base_quat"])
        normalizer["base_quat"] = get_identity_normalizer_from_stat(base_quat_stat)

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

        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        T_slice = slice(n_obs)

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = self.base_frame_obs_chunks[key][idx].astype(np.float32)
            del data[key]
        obs_dict["base_pos"] = self.base_frame_obs_chunks["base_pos"][idx].astype(np.float32)
        obs_dict["base_quat"] = self.base_frame_obs_chunks["base_quat"][idx].astype(np.float32)
        obs_dict["_obs_in_base_frame"] = np.ones((1,), dtype=np.float32)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(self.actions[idx].astype(np.float32)),
            "rel_action": torch.from_numpy(self.rel_actions[idx].astype(np.float32)),
        }
        return torch_data

    def _compute_rel_action_chunks_indexed(
        self,
        action_chunks: np.ndarray,
        left_pos_ref: np.ndarray,
        left_quat_ref: np.ndarray,
        right_pos_ref: np.ndarray,
        right_quat_ref: np.ndarray,
        batch_size: int = 2048,
    ) -> np.ndarray:
        n_samples, horizon, _ = action_chunks.shape
        rel_action_chunks = np.empty_like(action_chunks)

        for start in tqdm.tqdm(range(0, n_samples, batch_size), desc="Precomputing action chunks"):
            end = min(start + batch_size, n_samples)
            action = torch.from_numpy(action_chunks[start:end].astype(np.float32))
            batch_n = action.shape[0]
            eye = torch.eye(4, dtype=action.dtype)

            left_abs_T = eye.view(1, 1, 4, 4).repeat(batch_n, horizon, 1, 1)
            left_abs_T[:, :, :3, :3] = self.sixd_to_mat.forward(
                action[:, :, 3:9].reshape(-1, 6)
            ).reshape(batch_n, horizon, 3, 3)
            left_abs_T[:, :, :3, 3] = action[:, :, :3]

            right_abs_T = eye.view(1, 1, 4, 4).repeat(batch_n, horizon, 1, 1)
            right_abs_T[:, :, :3, :3] = self.sixd_to_mat.forward(
                action[:, :, 12:18].reshape(-1, 6)
            ).reshape(batch_n, horizon, 3, 3)
            right_abs_T[:, :, :3, 3] = action[:, :, 9:12]

            left_cur_rot = self.quat_to_mat.forward(
                torch.from_numpy(left_quat_ref[start:end].astype(np.float32))
            )
            left_cur_pos = torch.from_numpy(left_pos_ref[start:end].astype(np.float32))
            left_cur_T = eye.view(1, 1, 4, 4).repeat(batch_n, horizon, 1, 1)
            left_cur_T[:, :, :3, :3] = left_cur_rot.unsqueeze(1).expand(batch_n, horizon, 3, 3)
            left_cur_T[:, :, :3, 3] = left_cur_pos.unsqueeze(1).expand(batch_n, horizon, 3)

            right_cur_rot = self.quat_to_mat.forward(
                torch.from_numpy(right_quat_ref[start:end].astype(np.float32))
            )
            right_cur_pos = torch.from_numpy(right_pos_ref[start:end].astype(np.float32))
            right_cur_T = eye.view(1, 1, 4, 4).repeat(batch_n, horizon, 1, 1)
            right_cur_T[:, :, :3, :3] = right_cur_rot.unsqueeze(1).expand(batch_n, horizon, 3, 3)
            right_cur_T[:, :, :3, 3] = right_cur_pos.unsqueeze(1).expand(batch_n, horizon, 3)

            left_rel_T = torch.linalg.inv(left_cur_T) @ left_abs_T
            right_rel_T = torch.linalg.inv(right_cur_T) @ right_abs_T

            rel_action_chunks[start:end] = torch.cat(
                [
                    left_rel_T[:, :, :3, 3],
                    self.sixd_to_mat.inverse(left_rel_T[:, :, :3, :3].reshape(-1, 3, 3)).reshape(
                        batch_n, horizon, 6
                    ),
                    right_rel_T[:, :, :3, 3],
                    self.sixd_to_mat.inverse(right_rel_T[:, :, :3, :3].reshape(-1, 3, 3)).reshape(
                        batch_n, horizon, 6
                    ),
                    action[:, :, 18:],
                ],
                dim=-1,
            ).numpy()

        return rel_action_chunks

    def _precompute_action_chunks_loop(self):
        n_samples = len(self.sampler)
        action_dim = self.shape_meta["action"]["shape"][0]
        action_chunks = np.zeros((n_samples, self.horizon, action_dim), dtype=np.float32)
        rel_action_chunks = np.zeros_like(action_chunks)

        base_obs_idx = self.n_obs_steps - 1 if self.n_obs_steps else -1

        for i in tqdm.tqdm(range(n_samples), desc="Precomputing action chunks"):
            # sample = subset_sampler.sample_sequence(i)
            sample = self.sampler.sample_sequence(i)
            action_np = sample["action"].astype(np.float32)
            action_chunks[i] = action_np
            rel_action_np = _compute_rel_action_chunk(
                data_action=action_np,
                left_pos=sample["left_ee_pos"].astype(np.float32),
                left_quat=sample["left_ee_quat"].astype(np.float32),
                right_pos=sample["right_ee_pos"].astype(np.float32),
                right_quat=sample["right_ee_quat"].astype(np.float32),
                sixd_to_mat=self.sixd_to_mat,
                quat_to_mat=self.quat_to_mat,
                base_obs_idx=base_obs_idx,
            )
            rel_action_chunks[i] = rel_action_np

        return action_chunks, rel_action_chunks

    def _precompute_action_chunks(self):
        base_obs_idx = self.n_obs_steps - 1 if self.n_obs_steps else -1
        base_obs_idx = max(min(base_obs_idx, self.horizon - 1), 0)

        action_chunks = gather_padded_chunks_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer["action"]).astype(np.float32),
            key="action",
            out_length=self.horizon,
        ).astype(np.float32)
        left_pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer["left_ee_pos"]).astype(np.float32),
            key="left_ee_pos",
            timestep=base_obs_idx,
        ).astype(np.float32)
        left_quat_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer["left_ee_quat"]).astype(np.float32),
            key="left_ee_quat",
            timestep=base_obs_idx,
        ).astype(np.float32)
        right_pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer["right_ee_pos"]).astype(np.float32),
            key="right_ee_pos",
            timestep=base_obs_idx,
        ).astype(np.float32)
        right_quat_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer["right_ee_quat"]).astype(np.float32),
            key="right_ee_quat",
            timestep=base_obs_idx,
        ).astype(np.float32)
        rel_action_chunks = self._compute_rel_action_chunks_indexed(
            action_chunks=action_chunks,
            left_pos_ref=left_pos_ref,
            left_quat_ref=left_quat_ref,
            right_pos_ref=right_pos_ref,
            right_quat_ref=right_quat_ref,
        )
        return action_chunks, rel_action_chunks

    def _transform_obs_to_base_frame(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "base_pos" not in obs_dict or "base_quat" not in obs_dict:
            missing = []
            if "base_pos" not in obs_dict:
                missing.append("base_pos")
            if "base_quat" not in obs_dict:
                missing.append("base_quat")
            raise KeyError(
                f"Missing required key(s) for base-frame conversion: {', '.join(missing)}"
            )

        result = dict(obs_dict)
        base_pos = torch.from_numpy(obs_dict["base_pos"].astype(np.float32))
        base_quat = torch.from_numpy(obs_dict["base_quat"].astype(np.float32))
        base_rot = self.quat_to_mat.forward(base_quat)
        horizon = base_pos.shape[0]
        eye = torch.eye(4, dtype=base_pos.dtype)
        base_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
        base_T[:, :3, :3] = base_rot
        base_T[:, :3, 3] = base_pos
        base_T_inv = torch.linalg.inv(base_T)

        for key in ("left_ee_pos", "right_ee_pos", "head_site_pos"):
            if key not in result:
                continue
            pos = torch.from_numpy(result[key].astype(np.float32))
            pos_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
            pos_T[:, :3, 3] = pos
            rel_pos_T = base_T_inv @ pos_T
            result[key] = rel_pos_T[:, :3, 3].numpy().astype(np.float32)

        for key in ("left_ee_quat", "right_ee_quat", "head_site_quat"):
            if key not in result:
                continue
            quat = torch.from_numpy(result[key].astype(np.float32))
            rot = self.quat_to_mat.forward(quat)
            quat_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
            quat_T[:, :3, :3] = rot
            rel_quat_T = base_T_inv @ quat_T
            base_quat_key = self.mat_to_quat.forward(rel_quat_T[:, :3, :3])
            result[key] = base_quat_key.numpy().astype(np.float32)

        result["base_pos"] = obs_dict["base_pos"].astype(np.float32)
        result["base_quat"] = obs_dict["base_quat"].astype(np.float32)
        return result

    def _precompute_base_frame_lowdim(self) -> Dict[str, np.ndarray]:
        world_obs = {
            key: np.asarray(self.replay_buffer[key]).astype(np.float32) for key in self.lowdim_keys
        }
        result = self._transform_obs_to_base_frame(world_obs)
        self._precompute_base_frame_obs_chunks(base_frame_lowdim=result)
        return result

    def _precompute_base_frame_obs_chunks_loop(
        self, base_frame_lowdim: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        n_samples = len(self.sampler)
        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        obs_chunks = {}
        if base_frame_lowdim is None:
            base_frame_lowdim = self.base_frame_lowdim

        for key in self.lowdim_keys + ["base_pos", "base_quat"]:
            if key in self.lowdim_keys:
                shape = self.replay_buffer[key].shape[1:]
            else:
                shape = base_frame_lowdim[key].shape[1:]
            obs_chunks[key] = np.zeros((n_samples, n_obs) + shape, dtype=np.float32)

        for i in tqdm.tqdm(range(n_samples), desc="Precomputing base-frame obs chunks"):
            sample = self.sampler.sample_sequence(i)
            world_obs = {
                key: sample[key].astype(np.float32)
                for key in self.lowdim_keys
            }
            base_obs = self._transform_obs_to_base_frame(world_obs)
            for key in self.lowdim_keys + ["base_pos", "base_quat"]:
                obs_chunks[key][i] = base_obs[key][:n_obs].astype(np.float32)

        return obs_chunks

    def _precompute_base_frame_obs_chunks(
        self, base_frame_lowdim: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        if base_frame_lowdim is None:
            base_frame_lowdim = self.base_frame_lowdim

        obs_chunks = {}
        for key in self.lowdim_keys + ["base_pos", "base_quat"]:
            obs_chunks[key] = gather_padded_chunks_from_full_array(
                indices=self.sampler.indices,
                key_first_k=self.key_first_k,
                full_array=base_frame_lowdim[key],
                key=key,
                out_length=n_obs,
            ).astype(np.float32)

        self.base_frame_obs_chunks = obs_chunks
        return obs_chunks


def _compute_rel_action_chunk(
    data_action: np.ndarray,
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
    sixd_to_mat: RotationTransformer,
    quat_to_mat: RotationTransformer,
    base_obs_idx: int,
) -> np.ndarray:
    """
    Compute relative actions for a sampled trajectory chunk.
    Uses the last observed ee pose (indexed by base_obs_idx) as the reference frame
    for all future actions in the chunk.
    """
    action = torch.from_numpy(data_action)
    horizon = action.shape[0]
    eye = torch.eye(4, dtype=action.dtype)

    left_abs_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
    left_abs_T[:, :3, :3] = sixd_to_mat.forward(action[:, 3:9])
    left_abs_T[:, :3, 3] = action[:, :3]

    right_abs_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
    right_abs_T[:, :3, :3] = sixd_to_mat.forward(action[:, 12:18])
    right_abs_T[:, :3, 3] = action[:, 9:12]

    base_idx = base_obs_idx if base_obs_idx is not None else -1
    base_idx = max(min(base_idx, horizon - 1), 0)

    left_cur_rot = quat_to_mat.forward(
        torch.from_numpy(left_quat[base_idx : base_idx + 1])
    )
    left_cur_pos = torch.from_numpy(left_pos[base_idx : base_idx + 1])
    left_cur_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
    left_cur_T[:, :3, :3] = left_cur_rot.expand(horizon, -1, -1)
    left_cur_T[:, :3, 3] = left_cur_pos.expand(horizon, -1)

    right_cur_rot = quat_to_mat.forward(
        torch.from_numpy(right_quat[base_idx : base_idx + 1])
    )
    right_cur_pos = torch.from_numpy(right_pos[base_idx : base_idx + 1])
    right_cur_T = eye.unsqueeze(0).repeat(horizon, 1, 1)
    right_cur_T[:, :3, :3] = right_cur_rot.expand(horizon, -1, -1)
    right_cur_T[:, :3, 3] = right_cur_pos.expand(horizon, -1)

    left_rel_T = torch.linalg.inv(left_cur_T) @ left_abs_T
    right_rel_T = torch.linalg.inv(right_cur_T) @ right_abs_T

    left_rel_xyz = left_rel_T[:, :3, 3]
    left_rel_6d = sixd_to_mat.inverse(left_rel_T[:, :3, :3])
    right_rel_xyz = right_rel_T[:, :3, 3]
    right_rel_6d = sixd_to_mat.inverse(right_rel_T[:, :3, :3])

    rel_action = torch.cat(
        [
            left_rel_xyz,
            left_rel_6d,
            right_rel_xyz,
            right_rel_6d,
            action[:, 18:],
        ],
        dim=-1,
    )
    return rel_action.numpy()


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
        "base_pos": "obs_base_pos",
        "base_quat": "obs_base_quat",
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
