from typing import Dict, List
import copy
import os
import shutil

import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits

from mof.common.mof_transform_util import (
    action_rot6d_row_to_column,
    convert_base_pose_action_to_rel_trans,
    convert_base_pose_action_to_rel_traj,
    convert_world_action_to_frame,
    transform_obs_pose_dict_to_frame,
    world_frame_to_transform,
)
from mof.common.normalize_util import (
    array_to_stats,
    bigym_action_only_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from mof.common.pytorch_util import dict_apply
from mof.common.replay_buffer import ReplayBuffer
from mof.common.sampler import SequenceSampler, get_val_mask
from mof.dataset.base_dataset import BaseImageDataset
from mof.dataset.bigym_rby1_precompute_util import (
    gather_padded_chunks_from_full_array,
    gather_padded_timestep_from_full_array,
)
from mof.dataset.bigym_rby1_replay_dataset_base_rel_trans import (
    _convert_rby1_safetensors_to_replay,
)
from mof.model.common.normalizer import LinearNormalizer
from mof.model.common.rotation_transformer import RotationTransformer
from mof.codecs.imagecodecs_numcodecs import register_codecs

register_codecs(verbose=False)


class BigymRBY1ReplayDatasetMoF(BaseImageDataset):
    FRAME_PREFIXES = ("base", "left", "right")
    _ROT_CHANNELS = (3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17)
    _LEFT_POS_CHANNELS = (0, 1, 2)
    _RIGHT_POS_CHANNELS = (9, 10, 11)

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
        if use_cache:
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

        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
            for key in ("base_pos", "base_quat", "left_ee_pos", "left_ee_quat", "right_ee_pos", "right_ee_quat"):
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
        self.key_first_k = key_first_k
        self.quat_to_mat = RotationTransformer(from_rep="quaternion", to_rep="matrix")
        self.mat_to_quat = RotationTransformer(from_rep="matrix", to_rep="quaternion")
        self.base_frame_lowdim = self._precompute_base_frame_lowdim()
        self.left_frame_lowdim = self._precompute_side_frame_lowdim(frame_name="left")
        self.right_frame_lowdim = self._precompute_side_frame_lowdim(frame_name="right")
        self.actions = self._precompute_world_action_chunks()
        self.base_actions = self._precompute_frame_action_chunks(frame_name="base")
        self.left_actions = self._precompute_frame_action_chunks(frame_name="left")
        self.right_actions = self._precompute_frame_action_chunks(frame_name="right")
        self.base_rel_trans_actions = self._precompute_base_rel_trans_action_chunks()
        self.rel_traj_actions = self._precompute_rel_traj_action_chunks()
        self._precompute_frame_obs_chunks()
        self._precompute_world_ref_chunks()

    def _frame_key(self, frame_name: str, key: str) -> str:
        return f"{frame_name}__{key}"

    def _enforce_action_rot_identity_normalizer(self, normalizer: LinearNormalizer, key: str):
        scale = normalizer[key].params_dict["scale"]
        offset = normalizer[key].params_dict["offset"]
        scale.data[list(self._ROT_CHANNELS)] = 1.0
        offset.data[list(self._ROT_CHANNELS)] = 0.0

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k=self.key_first_k,
        )
        val_set.train_mask = ~self.train_mask
        val_set.actions = val_set._precompute_world_action_chunks()
        val_set.base_actions = val_set._precompute_frame_action_chunks(frame_name="base")
        val_set.left_actions = val_set._precompute_frame_action_chunks(frame_name="left")
        val_set.right_actions = val_set._precompute_frame_action_chunks(frame_name="right")
        val_set.base_rel_trans_actions = val_set._precompute_base_rel_trans_action_chunks()
        val_set.rel_traj_actions = val_set._precompute_rel_traj_action_chunks()
        val_set._precompute_frame_obs_chunks()
        val_set._precompute_world_ref_chunks()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Build range normalizers for all action and observation keys."""
        normalizer = LinearNormalizer()
        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = bigym_action_only_normalizer_from_stat(action_stat)

        expert_data = {
            "base": ("base_action", self.base_actions),
            "left": ("left_action", self.left_actions),
            "right": ("right_action", self.right_actions),
            "base_rel_trans": ("base_rel_trans_action", self.base_rel_trans_actions),
            "rel_traj": ("rel_traj_action", self.rel_traj_actions),
        }
        for name, (norm_key, data) in expert_data.items():
            stat = array_to_stats(data.reshape(-1, data.shape[-1]))
            normalizer[norm_key] = bigym_action_only_normalizer_from_stat(stat)

        for frame_name, frame_lowdim in (
            ("base", self.base_frame_lowdim),
            ("left", self.left_frame_lowdim),
            ("right", self.right_frame_lowdim),
        ):
            for key in self.lowdim_keys:
                if key.endswith("pos"):
                    stat = array_to_stats(frame_lowdim[key])
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif key.endswith("quat"):
                    stat = array_to_stats(frame_lowdim[key])
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                else:
                    stat = array_to_stats(frame_lowdim[key])
                    this_normalizer = get_range_normalizer_from_stat(stat)
                normalizer[self._frame_key(frame_name, key)] = this_normalizer

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

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[self._frame_key("base", key)] = self.base_frame_obs_chunks[key][idx].astype(np.float32)
            obs_dict[self._frame_key("left", key)] = self.left_frame_obs_chunks[key][idx].astype(np.float32)
            obs_dict[self._frame_key("right", key)] = self.right_frame_obs_chunks[key][idx].astype(np.float32)
            del data[key]

        obs_dict["base_pos"] = self.world_ref_obs_chunks["base_pos"][idx].astype(np.float32)
        obs_dict["base_quat"] = self.world_ref_obs_chunks["base_quat"][idx].astype(np.float32)
        obs_dict["_left_ref_pos_world"] = self.world_ref_obs_chunks["left_ee_pos"][idx].astype(np.float32)
        obs_dict["_left_ref_quat_world"] = self.world_ref_obs_chunks["left_ee_quat"][idx].astype(np.float32)
        obs_dict["_right_ref_pos_world"] = self.world_ref_obs_chunks["right_ee_pos"][idx].astype(np.float32)
        obs_dict["_right_ref_quat_world"] = self.world_ref_obs_chunks["right_ee_quat"][idx].astype(np.float32)

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(self.actions[idx].astype(np.float32)),
            "base_action": torch.from_numpy(self.base_actions[idx].astype(np.float32)),
            "left_action": torch.from_numpy(self.left_actions[idx].astype(np.float32)),
            "right_action": torch.from_numpy(self.right_actions[idx].astype(np.float32)),
            "base_rel_trans_action": torch.from_numpy(self.base_rel_trans_actions[idx].astype(np.float32)),
            "rel_traj_action": torch.from_numpy(self.rel_traj_actions[idx].astype(np.float32)),
        }

    def _world_lowdim_dict(self) -> Dict[str, np.ndarray]:
        return {key: np.asarray(self.replay_buffer[key]).astype(np.float32) for key in self.lowdim_keys}

    def _precompute_base_frame_lowdim(self) -> Dict[str, np.ndarray]:
        world_obs = self._world_lowdim_dict()
        base_T = world_frame_to_transform(
            pos=torch.from_numpy(np.asarray(self.replay_buffer["base_pos"]).astype(np.float32)),
            quat=torch.from_numpy(np.asarray(self.replay_buffer["base_quat"]).astype(np.float32)),
            quat_to_mat=self.quat_to_mat,
        )
        result = transform_obs_pose_dict_to_frame(
            obs_dict=dict_apply(world_obs, torch.from_numpy),
            world_T_frame=base_T,
            quat_to_mat=self.quat_to_mat,
            mat_to_quat=self.mat_to_quat,
        )
        return {key: value.numpy().astype(np.float32) for key, value in result.items()}

    def _precompute_side_frame_lowdim(self, frame_name: str) -> Dict[str, np.ndarray]:
        world_obs = self._world_lowdim_dict()
        if frame_name == "left":
            pos_key = "left_ee_pos"
            quat_key = "left_ee_quat"
        elif frame_name == "right":
            pos_key = "right_ee_pos"
            quat_key = "right_ee_quat"
        else:
            raise ValueError(f"Unsupported frame_name '{frame_name}'.")
        frame_T = world_frame_to_transform(
            pos=torch.from_numpy(np.asarray(self.replay_buffer[pos_key]).astype(np.float32)),
            quat=torch.from_numpy(np.asarray(self.replay_buffer[quat_key]).astype(np.float32)),
            quat_to_mat=self.quat_to_mat,
        )
        result = transform_obs_pose_dict_to_frame(
            obs_dict=dict_apply(world_obs, torch.from_numpy),
            world_T_frame=frame_T,
            quat_to_mat=self.quat_to_mat,
            mat_to_quat=self.mat_to_quat,
        )
        return {key: value.numpy().astype(np.float32) for key, value in result.items()}

    def _precompute_frame_obs_chunks(self):
        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        self.base_frame_obs_chunks = {}
        self.left_frame_obs_chunks = {}
        self.right_frame_obs_chunks = {}
        for key in self.lowdim_keys:
            self.base_frame_obs_chunks[key] = gather_padded_chunks_from_full_array(
                indices=self.sampler.indices,
                key_first_k=self.key_first_k,
                full_array=self.base_frame_lowdim[key],
                key=key,
                out_length=n_obs,
            ).astype(np.float32)
            self.left_frame_obs_chunks[key] = gather_padded_chunks_from_full_array(
                indices=self.sampler.indices,
                key_first_k=self.key_first_k,
                full_array=self.left_frame_lowdim[key],
                key=key,
                out_length=n_obs,
            ).astype(np.float32)
            self.right_frame_obs_chunks[key] = gather_padded_chunks_from_full_array(
                indices=self.sampler.indices,
                key_first_k=self.key_first_k,
                full_array=self.right_frame_lowdim[key],
                key=key,
                out_length=n_obs,
            ).astype(np.float32)

    def _precompute_world_ref_chunks(self):
        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        self.world_ref_obs_chunks = {}
        for key in ("base_pos", "base_quat", "left_ee_pos", "left_ee_quat", "right_ee_pos", "right_ee_quat"):
            self.world_ref_obs_chunks[key] = gather_padded_chunks_from_full_array(
                indices=self.sampler.indices,
                key_first_k=self.key_first_k,
                full_array=np.asarray(self.replay_buffer[key]).astype(np.float32),
                key=key,
                out_length=n_obs,
            ).astype(np.float32)

    def _precompute_world_action_chunks(self) -> np.ndarray:
        return gather_padded_chunks_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer["action"]).astype(np.float32),
            key="action",
            out_length=self.horizon,
        ).astype(np.float32)

    def _precompute_frame_action_chunks(self, frame_name: str) -> np.ndarray:
        base_obs_idx = self.n_obs_steps - 1 if self.n_obs_steps else -1
        base_obs_idx = max(min(base_obs_idx, self.horizon - 1), 0)
        if frame_name == "base":
            pos_key = "base_pos"
            quat_key = "base_quat"
        elif frame_name == "left":
            pos_key = "left_ee_pos"
            quat_key = "left_ee_quat"
        elif frame_name == "right":
            pos_key = "right_ee_pos"
            quat_key = "right_ee_quat"
        else:
            raise ValueError(f"Unsupported frame_name '{frame_name}'.")

        pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer[pos_key]).astype(np.float32),
            key=pos_key,
            timestep=base_obs_idx,
        ).astype(np.float32)
        quat_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=np.asarray(self.replay_buffer[quat_key]).astype(np.float32),
            key=quat_key,
            timestep=base_obs_idx,
        ).astype(np.float32)
        world_T_frame = world_frame_to_transform(
            pos=torch.from_numpy(pos_ref),
            quat=torch.from_numpy(quat_ref),
            quat_to_mat=self.quat_to_mat,
        )
        # self.actions is in pytorch3d row convention; convert to column
        # convention before frame transform (MoE pipeline uses column internally).
        actions_col = action_rot6d_row_to_column(torch.from_numpy(self.actions))
        frame_actions = convert_world_action_to_frame(
            action_world=actions_col,
            world_T_frame=world_T_frame,
        )
        return frame_actions.numpy().astype(np.float32)

    def _precompute_base_rel_trans_action_chunks(self) -> np.ndarray:
        base_obs_idx = self.n_obs_steps - 1 if self.n_obs_steps else -1
        base_obs_idx = max(min(base_obs_idx, self.horizon - 1), 0)
        left_base_pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=self.base_frame_lowdim["left_ee_pos"].astype(np.float32),
            key="left_ee_pos",
            timestep=base_obs_idx,
        ).astype(np.float32)
        right_base_pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=self.base_frame_lowdim["right_ee_pos"].astype(np.float32),
            key="right_ee_pos",
            timestep=base_obs_idx,
        ).astype(np.float32)
        rel_trans = convert_base_pose_action_to_rel_trans(
            base_action=torch.from_numpy(self.base_actions),
            left_base_pos_ref=torch.from_numpy(left_base_pos_ref)[:, None, :],
            right_base_pos_ref=torch.from_numpy(right_base_pos_ref)[:, None, :],
        )
        return rel_trans.numpy().astype(np.float32)

    def _precompute_rel_traj_action_chunks(self) -> np.ndarray:
        base_obs_idx = self.n_obs_steps - 1 if self.n_obs_steps else -1
        base_obs_idx = max(min(base_obs_idx, self.horizon - 1), 0)
        left_base_pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=self.base_frame_lowdim["left_ee_pos"].astype(np.float32),
            key="left_ee_pos",
            timestep=base_obs_idx,
        ).astype(np.float32)
        left_base_quat_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=self.base_frame_lowdim["left_ee_quat"].astype(np.float32),
            key="left_ee_quat",
            timestep=base_obs_idx,
        ).astype(np.float32)
        right_base_pos_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=self.base_frame_lowdim["right_ee_pos"].astype(np.float32),
            key="right_ee_pos",
            timestep=base_obs_idx,
        ).astype(np.float32)
        right_base_quat_ref = gather_padded_timestep_from_full_array(
            indices=self.sampler.indices,
            key_first_k=self.key_first_k,
            full_array=self.base_frame_lowdim["right_ee_quat"].astype(np.float32),
            key="right_ee_quat",
            timestep=base_obs_idx,
        ).astype(np.float32)
        rel_traj = convert_base_pose_action_to_rel_traj(
            base_action=torch.from_numpy(self.base_actions),
            left_base_pos_ref=torch.from_numpy(left_base_pos_ref)[:, None, :],
            left_base_quat_ref=torch.from_numpy(left_base_quat_ref)[:, None, :],
            right_base_pos_ref=torch.from_numpy(right_base_pos_ref)[:, None, :],
            right_base_quat_ref=torch.from_numpy(right_base_quat_ref)[:, None, :],
            quat_to_mat=self.quat_to_mat,
        )
        return rel_traj.numpy().astype(np.float32)
