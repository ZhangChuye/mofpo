"""Ablation 1: Gram-Schmidt pose+epsilon transform (column convention).

Stock MoF ([mixture_of_frames_policy.py]) transforms rot6d blocks
with the ``*_as_vectors`` / ``rotate_raw_vectors`` family, i.e. it rotates the
two rot6d 3-vectors per arm *raw* (R @ vec) and skips Gram-Schmidt.

This subclass routes **every** rot6d frame transform through the matrix path
(``rotation_6d_columns_to_matrix`` -> Gram-Schmidt -> R @ matrix ->
``matrix_to_rotation_6d_columns``), orthogonalising the rot6d block before the
transform and re-encoding the first two **columns**. This now covers BOTH:
  - the pose (x_t) path  (sample frame conversions), and
  - the epsilon/delta path (per-expert eps -> canonical, native eps targets),
so the whole pipeline is Gram-Schmidt-consistent (no raw-vector / GS mix).

Row convention was the ideal per the request, but the whole MoE stack is
column-internal and there is no row-convention rel_traj matrix converter; per
the user's explicit fallback we use columns.

Transform substitutions
-----------------------
Pose path (affine: position gets R + translation):
  convert_pose_action_between_frames_as_vectors -> convert_pose_action_between_frames
  convert_base_pose_action_to_rel_traj_as_vectors -> convert_base_pose_action_to_rel_traj
  convert_rel_traj_to_base_pose_action_as_vectors -> convert_rel_traj_to_base_pose_action
Epsilon/delta path (linear: position gets R only, NO translation):
  rotate_raw_vectors_between_frames -> convert_pose_delta_between_frames
  convert_base_delta_to_rel_traj    -> _gs_delta_base_to_rel_traj  (*)
  convert_rel_traj_delta_to_base    -> _gs_delta_rel_traj_to_base   (*)
  (*) = the pose rel_traj converter called with ZERO position refs: zero ref
        removes the affine translation (delta-correct) while rot6d still flows
        through Gram-Schmidt. base / base_rel_trans eps apply no frame
        rotation (ref offset cancels in a delta) so they are unchanged.

DIFFERS FROM BASE in seven verbatim-copied orchestration methods (each with
the rot6d calls replaced as above, marked ``# GS:``):
  pose:  _canonical_sample_norm_to_base_pose_real, _build_expert_inputs,
         predict_action
  eps:   _convert_expert_output_to_canonical_norm,
         _base_noise_real_to_canonical_norm,
         _canonical_noise_norm_to_base_delta_real,
         _compute_native_expert_losses
No edits to the base policy.
"""
from typing import Dict, Tuple

import torch

from mof.common.mof_transform_util import (
    action_rot6d_column_to_row,
    convert_base_pose_action_to_rel_trans,
    convert_base_pose_action_to_rel_traj,            # GS pose counterpart
    convert_base_rel_trans_to_pose_action,
    convert_frame_action_to_world,
    convert_pose_action_between_frames,              # GS pose counterpart
    convert_pose_delta_between_frames,               # GS delta counterpart (L/R)
    convert_rel_traj_to_base_pose_action,            # GS pose / GS-delta-via-zero-ref
    eye_transform,
    normalize_like_delta,
    unnormalize_like_delta,
)
from mof.policy.mixture_of_frames_policy import (
    MixtureOfFramesPolicy,
)


def _gs_delta_base_to_rel_traj(delta, refs, quat_to_mat):
    """ε: base -> rel_traj, rot6d via Gram-Schmidt (column).

    Position rotated by R_ref^T with NO translation (delta-correct: the
    constant per-arm ref offset cancels in a delta). Implemented as the pose
    rel_traj converter with ZERO position refs -> (a) drops the affine
    translation, (b) routes rot6d through rotation_6d_columns_to_matrix
    (Gram-Schmidt) -> R_ref^T @ R -> first two columns.
    """
    z_l = torch.zeros_like(refs["left_pos"]).to(delta.dtype)
    z_r = torch.zeros_like(refs["right_pos"]).to(delta.dtype)
    return convert_base_pose_action_to_rel_traj(
        delta,
        z_l, refs["left_quat"].to(delta.dtype),
        z_r, refs["right_quat"].to(delta.dtype),
        quat_to_mat,
    )


def _gs_delta_rel_traj_to_base(delta, refs, quat_to_mat):
    """ε: rel_traj -> base, rot6d via Gram-Schmidt (column).

    Inverse of _gs_delta_base_to_rel_traj. Position rotated by R_ref with NO
    translation (zero position refs); rot6d through Gram-Schmidt.
    """
    z_l = torch.zeros_like(refs["left_pos"]).to(delta.dtype)
    z_r = torch.zeros_like(refs["right_pos"]).to(delta.dtype)
    return convert_rel_traj_to_base_pose_action(
        delta,
        z_l, refs["left_quat"].to(delta.dtype),
        z_r, refs["right_quat"].to(delta.dtype),
        quat_to_mat,
    )


class MixtureOfFramesPolicyGramSchmidt(MixtureOfFramesPolicy):
    """MoF with Gram-Schmidt (column) rot6d transforms on BOTH the x_t
    pose path and the epsilon/delta path."""

    # ================================================================== #
    # POSE PATH
    # ================================================================== #

    # canonical x_t (norm) -> base-frame pose (real)
    def _canonical_sample_norm_to_base_pose_real(
        self,
        canonical_sample_norm: torch.Tensor,
        base_frame_obs_raw: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._canonical_space == "base":
            return self.normalizer["base_action"].unnormalize(canonical_sample_norm)
        if self._canonical_space in ("left", "right"):
            canonical_real = self.normalizer[f"{self._canonical_space}_action"].unnormalize(
                canonical_sample_norm
            )
            base_T_frame = self._base_T_ee_frame(
                self._canonical_space, base_frame_obs_raw
            ).to(dtype=canonical_sample_norm.dtype)
            eye_T = eye_transform(
                canonical_sample_norm.shape[0], 1,
                dtype=canonical_sample_norm.dtype, device=canonical_sample_norm.device,
            )
            # GS: convert_pose_action_between_frames_as_vectors -> matrix path
            return convert_pose_action_between_frames(
                action=canonical_real,
                world_T_source_frame=base_T_frame,
                world_T_target_frame=eye_T,
            )
        refs = self._get_ee_refs(base_frame_obs_raw)
        if self._canonical_space == "base_rel_trans":
            canonical_real = self.normalizer["base_rel_trans_action"].unnormalize(canonical_sample_norm)
            # pos-only (no rot6d transform) -> identical to base
            return convert_base_rel_trans_to_pose_action(
                canonical_real, refs["left_pos"], refs["right_pos"]
            )
        canonical_real = self.normalizer["rel_traj_action"].unnormalize(canonical_sample_norm)
        # GS: convert_rel_traj_to_base_pose_action_as_vectors -> matrix path
        return convert_rel_traj_to_base_pose_action(
            canonical_real,
            refs["left_pos"],
            refs["left_quat"],
            refs["right_pos"],
            refs["right_quat"],
            self.quat_to_mat,
        )

    # base-frame pose (real) -> per-expert normalized inputs
    def _build_expert_inputs(
        self,
        canonical_sample_norm: torch.Tensor,
        frame_obs_norm: Dict[str, Dict[str, torch.Tensor]],
        rgb_features: Dict[str, torch.Tensor],
        obs_dict: Dict[str, torch.Tensor],
        base_frame_obs_raw: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        refs = {
            key: value.to(dtype=canonical_sample_norm.dtype)
            for key, value in self._get_ee_refs(base_frame_obs_raw).items()
        }

        world_T_base = self._frame_world_transform_from_obs(obs_dict, "base")

        base_pose_real = self._canonical_sample_norm_to_base_pose_real(
            canonical_sample_norm, base_frame_obs_raw,
        )
        expert_inputs = {}
        if "base" in self.model.enabled_experts:
            if self._canonical_space == "base":
                _base_expert_sample = canonical_sample_norm
            else:
                _base_expert_sample = self.normalizer["base_action"].normalize(base_pose_real)
            expert_inputs["base"] = {
                "sample": _base_expert_sample,
                "rgb_features": rgb_features,
                "obs_dict": frame_obs_norm["base"],
            }
        if "left" in self.model.enabled_experts:
            if self._canonical_space == "left":
                _left_sample = canonical_sample_norm
            else:
                world_T_left = self._frame_world_transform_from_obs(obs_dict, "left")
                # GS: convert_pose_action_between_frames_as_vectors -> matrix path
                left_sample_real = convert_pose_action_between_frames(
                    action=base_pose_real,
                    world_T_source_frame=world_T_base,
                    world_T_target_frame=world_T_left,
                )
                _left_sample = self.normalizer["left_action"].normalize(left_sample_real)
            expert_inputs["left"] = {
                "sample": _left_sample,
                "rgb_features": rgb_features,
                "obs_dict": frame_obs_norm["left"],
            }
        if "right" in self.model.enabled_experts:
            if self._canonical_space == "right":
                _right_sample = canonical_sample_norm
            else:
                world_T_right = self._frame_world_transform_from_obs(obs_dict, "right")
                # GS: convert_pose_action_between_frames_as_vectors -> matrix path
                right_sample_real = convert_pose_action_between_frames(
                    action=base_pose_real,
                    world_T_source_frame=world_T_base,
                    world_T_target_frame=world_T_right,
                )
                _right_sample = self.normalizer["right_action"].normalize(right_sample_real)
            expert_inputs["right"] = {
                "sample": _right_sample,
                "rgb_features": rgb_features,
                "obs_dict": frame_obs_norm["right"],
            }
        if "base_rel_trans" in self.model.enabled_experts:
            if self._canonical_space == "base_rel_trans":
                _rt_sample = canonical_sample_norm
            else:
                # pos-only (rot6d untouched) -> identical to base
                rel_trans_real = convert_base_pose_action_to_rel_trans(
                    base_pose_real,
                    left_base_pos_ref=refs["left_pos"],
                    right_base_pos_ref=refs["right_pos"],
                )
                _rt_sample = self.normalizer["base_rel_trans_action"].normalize(rel_trans_real)
            expert_inputs["base_rel_trans"] = {
                "sample": _rt_sample,
                "rgb_features": rgb_features,
                "obs_dict": frame_obs_norm["base"],
            }
        if "rel_traj" in self.model.enabled_experts:
            if self._canonical_space == "rel_traj":
                _rel_traj_sample = canonical_sample_norm
            else:
                # GS: convert_base_pose_action_to_rel_traj_as_vectors -> matrix path
                rel_traj_real = convert_base_pose_action_to_rel_traj(
                    base_pose_real,
                    refs["left_pos"],
                    refs["left_quat"],
                    refs["right_pos"],
                    refs["right_quat"],
                    self.quat_to_mat,
                )
                _rel_traj_sample = self.normalizer["rel_traj_action"].normalize(rel_traj_real)
            expert_inputs["rel_traj"] = {
                "sample": _rel_traj_sample,
                "rgb_features": rgb_features,
                "obs_dict": frame_obs_norm["base"],
            }
        return expert_inputs

    # inference: canonical sample -> base -> world
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        if self._is_batch_obs(obs_dict):
            frame_obs_raw = self._prepare_batch_frame_obs(obs_dict)
        else:
            frame_obs_raw = self._prepare_runtime_frame_obs(obs_dict)
        frame_obs_norm = {
            frame_name: self._normalize_frame_obs(frame_obs_raw[frame_name], frame_name)
            for frame_name in self.FRAME_NAMES
        }
        rgb_features = self.obs_encoder.forward_rgb(frame_obs_norm["base"])

        value = next(iter(frame_obs_norm["base"].values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            frame_obs_norm=frame_obs_norm,
            rgb_features=rgb_features,
            obs_dict=obs_dict,
            base_frame_obs_raw=frame_obs_raw["base"],
            **self.kwargs,
        )
        canonical_action = self.normalizer[self._train_action_key].unnormalize(nsample)
        if self._canonical_space == "base_rel_trans":
            refs = self._get_ee_refs(frame_obs_raw["base"])
            base_action = convert_base_rel_trans_to_pose_action(
                canonical_action,
                refs["left_pos"].to(canonical_action.dtype),
                refs["right_pos"].to(canonical_action.dtype),
            )
        elif self._canonical_space == "rel_traj":
            refs = self._get_ee_refs(frame_obs_raw["base"])
            # GS: convert_rel_traj_to_base_pose_action_as_vectors -> matrix path
            base_action = convert_rel_traj_to_base_pose_action(
                canonical_action,
                refs["left_pos"].to(canonical_action.dtype),
                refs["left_quat"].to(canonical_action.dtype),
                refs["right_pos"].to(canonical_action.dtype),
                refs["right_quat"].to(canonical_action.dtype),
                self.quat_to_mat,
            )
        elif self._canonical_space in ("left", "right"):
            world_T_canonical = self._frame_world_transform_from_obs(
                obs_dict, self._canonical_space
            ).to(dtype=canonical_action.dtype)
            world_T_base = self._frame_world_transform_from_obs(obs_dict, "base").to(
                dtype=canonical_action.dtype
            )
            # GS: convert_pose_action_between_frames_as_vectors -> matrix path
            base_action = convert_pose_action_between_frames(
                action=canonical_action,
                world_T_source_frame=world_T_canonical,
                world_T_target_frame=world_T_base,
            )
        else:
            base_action = canonical_action
        abs_action = convert_frame_action_to_world(
            action_frame=base_action,
            world_T_frame=self._frame_world_transform_from_obs(obs_dict, "base"),
        )
        abs_action = action_rot6d_column_to_row(abs_action)
        abs_action = self._clamp_world_action(abs_action)
        start = To - 1
        end = start + self.n_action_steps
        return {"action": abs_action[:, start:end], "action_pred": abs_action}

    # ================================================================== #
    # EPSILON / DELTA PATH  (rot6d through Gram-Schmidt; pos rotation-only)
    # ================================================================== #

    # expert epsilon (norm, native frame) -> canonical (norm)
    def _convert_expert_output_to_canonical_norm(
        self,
        expert_output_norm: torch.Tensor,
        expert_name: str,
        obs_dict: Dict[str, torch.Tensor],
        base_frame_obs_raw: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        world_T_base = self._frame_world_transform_from_obs(obs_dict, "base")
        if expert_name == "base":
            base_noise_real = unnormalize_like_delta(
                expert_output_norm, self.normalizer["base_action"]
            )
        elif expert_name in ("left", "right"):
            world_T_frame = self._frame_world_transform_from_obs(obs_dict, expert_name)
            frame_real = unnormalize_like_delta(
                expert_output_norm, self.normalizer[f"{expert_name}_action"]
            )
            # GS: rotate_raw_vectors_between_frames -> convert_pose_delta_between_frames
            base_noise_real = convert_pose_delta_between_frames(
                frame_real, world_T_frame, world_T_base
            )
        elif expert_name == "base_rel_trans":
            # no frame rotation (ref offset cancels in a delta) -> unchanged
            base_noise_real = unnormalize_like_delta(
                expert_output_norm, self.normalizer["base_rel_trans_action"]
            )
        elif expert_name == "rel_traj":
            refs = self._get_ee_refs(base_frame_obs_raw)
            rel_traj_noise_real = unnormalize_like_delta(
                expert_output_norm, self.normalizer["rel_traj_action"]
            )
            # GS: convert_rel_traj_delta_to_base -> GS-delta via zero pos refs
            base_noise_real = _gs_delta_rel_traj_to_base(
                rel_traj_noise_real, refs, self.quat_to_mat
            )
        else:
            raise ValueError(f"Unsupported expert '{expert_name}'.")
        return self._base_noise_real_to_canonical_norm(base_noise_real, base_frame_obs_raw)

    # base-frame noise (real) -> canonical (norm)
    def _base_noise_real_to_canonical_norm(
        self, base_noise_real: torch.Tensor, base_frame_obs_raw: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self._canonical_space in ("base", "base_rel_trans"):
            return normalize_like_delta(base_noise_real, self.normalizer[self._train_action_key])
        if self._canonical_space in ("left", "right"):
            base_T_frame = self._base_T_ee_frame(
                self._canonical_space, base_frame_obs_raw
            ).to(dtype=base_noise_real.dtype)
            eye_T = eye_transform(
                base_noise_real.shape[0], 1,
                dtype=base_noise_real.dtype, device=base_noise_real.device,
            )
            # GS: rotate_raw_vectors_between_frames -> convert_pose_delta_between_frames
            frame_noise_real = convert_pose_delta_between_frames(
                base_noise_real,
                world_T_source_frame=eye_T,
                world_T_target_frame=base_T_frame,
            )
            return normalize_like_delta(
                frame_noise_real, self.normalizer[f"{self._canonical_space}_action"]
            )
        refs = self._get_ee_refs(base_frame_obs_raw)
        # GS: convert_base_delta_to_rel_traj -> GS-delta via zero pos refs
        rel_traj_noise_real = _gs_delta_base_to_rel_traj(
            base_noise_real, refs, self.quat_to_mat
        )
        return normalize_like_delta(rel_traj_noise_real, self.normalizer["rel_traj_action"])

    # canonical noise (norm) -> base-frame delta (real)  [inverse of above]
    def _canonical_noise_norm_to_base_delta_real(
        self, canonical_noise_norm: torch.Tensor, base_frame_obs_raw: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self._canonical_space in ("base", "base_rel_trans"):
            return unnormalize_like_delta(canonical_noise_norm, self.normalizer[self._train_action_key])
        if self._canonical_space in ("left", "right"):
            frame_noise_real = unnormalize_like_delta(
                canonical_noise_norm, self.normalizer[f"{self._canonical_space}_action"]
            )
            base_T_frame = self._base_T_ee_frame(
                self._canonical_space, base_frame_obs_raw
            ).to(dtype=canonical_noise_norm.dtype)
            eye_T = eye_transform(
                canonical_noise_norm.shape[0], 1,
                dtype=canonical_noise_norm.dtype, device=canonical_noise_norm.device,
            )
            # GS: rotate_raw_vectors_between_frames -> convert_pose_delta_between_frames
            return convert_pose_delta_between_frames(
                frame_noise_real,
                world_T_source_frame=base_T_frame,
                world_T_target_frame=eye_T,
            )
        refs = self._get_ee_refs(base_frame_obs_raw)
        rel_traj_noise_real = unnormalize_like_delta(
            canonical_noise_norm, self.normalizer["rel_traj_action"]
        )
        # GS: convert_rel_traj_delta_to_base -> GS-delta via zero pos refs
        return _gs_delta_rel_traj_to_base(
            rel_traj_noise_real, refs, self.quat_to_mat
        )

    # per-expert native epsilon loss (each expert's own frame)
    def _compute_native_expert_losses(
        self,
        model_out: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        loss_mask: torch.Tensor,
        batch_obs_repeated: Dict[str, torch.Tensor],
        frame_obs_raw_base_repeated: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        losses = {}
        batch_losses = {}
        total = noise.new_zeros(())
        world_T_base = self._frame_world_transform_from_obs(batch_obs_repeated, "base")
        base_noise_real = self._canonical_noise_norm_to_base_delta_real(
            noise, frame_obs_raw_base_repeated
        )
        refs = self._get_ee_refs(frame_obs_raw_base_repeated)
        for name in model_out["expert_names"]:
            expert_pred = model_out["expert_outputs"][name]
            if name == "base_rel_trans":
                action_key = "base_rel_trans_action"
            elif name == "rel_traj":
                action_key = "rel_traj_action"
            else:
                action_key = f"{name}_action"
            if name in ("left", "right"):
                world_T_expert = self._frame_world_transform_from_obs(batch_obs_repeated, name)
                # GS: rotate_raw_vectors_between_frames -> convert_pose_delta_between_frames
                expert_noise_real = convert_pose_delta_between_frames(
                    base_noise_real, world_T_base, world_T_expert
                )
                expert_target = normalize_like_delta(expert_noise_real, self.normalizer[action_key])
            elif name in ("base", "base_rel_trans"):
                # no frame rotation -> unchanged
                expert_target = normalize_like_delta(base_noise_real, self.normalizer[action_key])
            elif name == "rel_traj":
                # GS: convert_base_delta_to_rel_traj -> GS-delta via zero pos refs
                expert_noise_real = _gs_delta_base_to_rel_traj(
                    base_noise_real, refs, self.quat_to_mat
                )
                expert_target = normalize_like_delta(expert_noise_real, self.normalizer[action_key])
            else:
                raise ValueError(f"Unsupported expert '{name}'.")
            batch_loss = self._masked_batch_mse(expert_pred, expert_target, loss_mask)
            batch_losses[name] = batch_loss
            loss = batch_loss.mean()
            losses[name] = loss
            total = total + loss
        total = total / len(model_out["expert_names"])
        return losses, batch_losses, total
