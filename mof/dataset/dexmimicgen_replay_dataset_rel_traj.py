from typing import Dict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from mof.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    robosuite_dual_arm_action_normalizer_from_stat,
    robosuite_dual_arm_action_normalizer_dex_from_stat,
)
from mof.common.pytorch_util import dict_apply
from mof.dataset.bigym_rby1_replay_dataset_rel_traj import (
    BigymRBY1ReplayDatasetRelTraj,
)
from mof.dataset.dexmimicgen_replay_dataset_mof import (
    DexMimicGenReplayDatasetMoF,
)
from mof.model.common.rotation_transformer import RotationTransformer
from mof.model.common.normalizer import LinearNormalizer


class DexMimicGenReplayDatasetRelTraj(DexMimicGenReplayDatasetMoF):
    """Standalone rel-traj dataset wrapper on top of DexMimicGen frame precompute.

    This reuses the same replay-buffer loading as the MoF dataset, but it
    must expose the *standalone* rel-traj contract expected by
    `DiffusionUnetPolicyRelTraj`: base-frame observations plus the historical
    per-arm local `rel_action` target used by the old Bigym rel-traj dataset.
    """

    _compute_rel_action_chunks_indexed = (
        BigymRBY1ReplayDatasetRelTraj._compute_rel_action_chunks_indexed
    )
    _precompute_action_chunks = BigymRBY1ReplayDatasetRelTraj._precompute_action_chunks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sixd_to_mat = RotationTransformer(
            from_rep="rotation_6d", to_rep="matrix"
        )
        _, self.rel_actions = self._precompute_action_chunks()

    def get_validation_dataset(self):
        val_set = super().get_validation_dataset()
        _, val_set.rel_actions = val_set._precompute_action_chunks()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Dispatch on grip_dim: action_dim=20 (parallel, grip_dim=1) vs
        # action_dim=30 (dex hand, grip_dim=6). Mirrors the mof dataset.
        action_dim = self.replay_buffer["action"].shape[-1]
        grip_dim = (action_dim - 18) // 2
        action_normalizer_fn = (
            robosuite_dual_arm_action_normalizer_from_stat
            if grip_dim == 1
            else robosuite_dual_arm_action_normalizer_dex_from_stat
        )

        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = action_normalizer_fn(action_stat)

        rel_action_stat = array_to_stats(
            self.rel_actions.reshape(-1, self.rel_actions.shape[-1])
        )
        normalizer["rel_action"] = action_normalizer_fn(rel_action_stat)

        for key in self.lowdim_keys:
            stat = array_to_stats(self.base_frame_lowdim[key])
            if key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                this_normalizer = get_range_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        base_pos_stat = array_to_stats(
            np.asarray(self.replay_buffer["base_pos"]).astype(np.float32)
        )
        normalizer["base_pos"] = get_range_normalizer_from_stat(base_pos_stat)
        base_quat_stat = array_to_stats(
            np.asarray(self.replay_buffer["base_quat"]).astype(np.float32)
        )
        normalizer["base_quat"] = get_identity_normalizer_from_stat(base_quat_stat)

        for key in self.rgb_keys:
            normalizer[key] = (
                get_image_range_normalizer()
                if self.normalize_rgb
                else get_image_identity_normalizer()
            )
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        n_obs = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        t_slice = slice(n_obs)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][t_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = self.base_frame_obs_chunks[key][idx].astype(np.float32)
            del data[key]
        obs_dict["base_pos"] = self.base_frame_obs_chunks["base_pos"][idx].astype(
            np.float32
        )
        obs_dict["base_quat"] = self.base_frame_obs_chunks["base_quat"][idx].astype(
            np.float32
        )
        obs_dict["_obs_in_base_frame"] = np.ones((1,), dtype=np.float32)

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(self.actions[idx].astype(np.float32)),
            "rel_action": torch.from_numpy(self.rel_actions[idx].astype(np.float32)),
        }
