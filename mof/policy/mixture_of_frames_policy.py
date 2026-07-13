from typing import Dict, Tuple, Union
import copy
import inspect

import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mof.common.mof_transform_util import (
    arm_channels,
    action_rot6d_column_to_row,
    convert_base_delta_to_rel_traj,
    convert_base_pose_action_to_rel_trans,
    convert_base_pose_action_to_rel_traj_as_vectors,
    convert_base_rel_trans_to_pose_action,
    convert_frame_action_to_world,
    convert_pose_action_between_frames_as_vectors,
    convert_rel_traj_delta_to_base,
    convert_rel_traj_to_base_pose_action_as_vectors,
    eye_transform,
    normalize_like_delta,
    rotate_raw_vectors_between_frames,
    transform_obs_pose_dict_to_frame,
    unnormalize_like_delta,
    world_frame_to_transform,
)
from mof.model.common.normalizer import LinearNormalizer
from mof.model.common.rotation_transformer import RotationTransformer
from mof.model.diffusion.conditional_unet1d_mof import ConditionalUnet1DMoF
from mof.model.diffusion.mask_generator import LowdimMaskGenerator
from mof.model.vision.timm_obs_encoder import TimmObsEncoder
from mof.policy.base_image_policy import BaseImagePolicy
from mof.policy.train_diffusion_utils import (
    repeat_train_diffusion_dict,
    repeat_train_diffusion_tensor,
)


class MixtureOfFramesPolicy(BaseImagePolicy):
    FRAME_NAMES = ("base", "left", "right")

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: TimmObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        train_diffusion_n_samples=1,
        enabled_experts=("base", "left", "right"),
        router_mode: str = "learned",
        router_hidden_dim: int = 256,
        router_timestep_embed_dim: int = 128,
        expert_loss_coef: float = 0.0,
        canonical_space: str = "base",
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond, "This MoF UNet currently supports obs_as_global_cond=True only."
        if canonical_space not in ("base", "left", "right", "base_rel_trans", "rel_traj"):
            raise ValueError(
                f"Unsupported canonical_space '{canonical_space}'. "
                f"Expected 'base', 'left', 'right', 'base_rel_trans', or 'rel_traj'."
            )
        self._left_ref_pos_world_key = "_left_ref_pos_world"
        self._left_ref_quat_world_key = "_left_ref_quat_world"
        self._right_ref_pos_world_key = "_right_ref_pos_world"
        self._right_ref_quat_world_key = "_right_ref_quat_world"
        self._recon_only_obs_keys = (
            "base_pos",
            "base_quat",
            self._left_ref_pos_world_key,
            self._left_ref_quat_world_key,
            self._right_ref_pos_world_key,
            self._right_ref_quat_world_key,
        )

        if hasattr(obs_encoder, "low_dim_keys"):
            obs_encoder.low_dim_keys = [
                key for key in obs_encoder.low_dim_keys if key not in ("base_pos", "base_quat")
            ]

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        low_dim_cond_dim = 0
        for key in getattr(obs_encoder, "low_dim_keys", []):
            if key not in shape_meta["obs"]:
                continue
            obs_attr = shape_meta["obs"][key]
            horizon_len = int(obs_attr["horizon"])
            feature_dim = 1
            for dim in obs_attr["shape"]:
                feature_dim *= int(dim)
            low_dim_cond_dim += horizon_len * feature_dim

        self.obs_encoder = obs_encoder
        self.shape_meta = shape_meta
        self.model = ConditionalUnet1DMoF(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            low_dim_cond_dim=low_dim_cond_dim,
            horizon=horizon,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            rgb_keys=getattr(obs_encoder, "rgb_keys", []),
            low_dim_keys=getattr(obs_encoder, "low_dim_keys", []),
            enabled_experts=enabled_experts,
            router_mode=router_mode,
            router_hidden_dim=router_hidden_dim,
            router_timestep_embed_dim=router_timestep_embed_dim,
        )
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.train_diffusion_n_samples = train_diffusion_n_samples
        self.kwargs = self._filter_scheduler_step_kwargs(kwargs)

        self._canonical_space = canonical_space
        if canonical_space == "base":
            self._train_action_key = "base_action"
        elif canonical_space == "left":
            self._train_action_key = "left_action"
        elif canonical_space == "right":
            self._train_action_key = "right_action"
        elif canonical_space == "base_rel_trans":
            self._train_action_key = "base_rel_trans_action"
        else:
            self._train_action_key = "rel_traj_action"
        self.router_mode = router_mode
        self.expert_loss_coef = float(expert_loss_coef)
        self._last_train_metrics = dict()

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.quat_to_mat = RotationTransformer(from_rep="quaternion", to_rep="matrix")
        self.mat_to_quat = RotationTransformer(from_rep="matrix", to_rep="quaternion")

    def _filter_scheduler_step_kwargs(self, kwargs: Dict) -> Dict:
        step_signature = inspect.signature(self.noise_scheduler.step)
        reserved = {"model_output", "timestep", "sample", "generator"}
        allowed = set(step_signature.parameters.keys()) - reserved
        return {key: value for key, value in kwargs.items() if key in allowed}

    def _frame_obs_key(self, frame_name: str, key: str) -> str:
        return f"{frame_name}__{key}"

    def _normalize_frame_obs(self, frame_obs: Dict[str, torch.Tensor], frame_name: str) -> Dict[str, torch.Tensor]:
        out = {}
        for key, value in frame_obs.items():
            if key in getattr(self.obs_encoder, "rgb_keys", []):
                out[key] = self.normalizer[key].normalize(value)
            else:
                out[key] = self.normalizer[self._frame_obs_key(frame_name, key)].normalize(value)
        return out

    def _extract_frame_obs_from_prefixed(self, obs_dict: Dict[str, torch.Tensor], frame_name: str) -> Dict[str, torch.Tensor]:
        frame_obs = {}
        for key in getattr(self.obs_encoder, "rgb_keys", []):
            frame_obs[key] = obs_dict[key]
        for key in getattr(self.obs_encoder, "low_dim_keys", []):
            frame_obs[key] = obs_dict[self._frame_obs_key(frame_name, key)]
        return frame_obs

    def _prepare_batch_frame_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {frame_name: self._extract_frame_obs_from_prefixed(obs_dict, frame_name) for frame_name in self.FRAME_NAMES}

    def _prepare_runtime_frame_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        world_obs = {
            key: value
            for key, value in obs_dict.items()
            if key in getattr(self.obs_encoder, "low_dim_keys", [])
        }
        base_T = world_frame_to_transform(obs_dict["base_pos"], obs_dict["base_quat"], self.quat_to_mat)
        left_T = world_frame_to_transform(obs_dict["left_ee_pos"], obs_dict["left_ee_quat"], self.quat_to_mat)
        right_T = world_frame_to_transform(obs_dict["right_ee_pos"], obs_dict["right_ee_quat"], self.quat_to_mat)
        frame_obs = {
            "base": transform_obs_pose_dict_to_frame(world_obs, base_T, self.quat_to_mat, self.mat_to_quat),
            "left": transform_obs_pose_dict_to_frame(world_obs, left_T, self.quat_to_mat, self.mat_to_quat),
            "right": transform_obs_pose_dict_to_frame(world_obs, right_T, self.quat_to_mat, self.mat_to_quat),
        }
        for frame_name in self.FRAME_NAMES:
            for key in getattr(self.obs_encoder, "rgb_keys", []):
                frame_obs[frame_name][key] = obs_dict[key]
        return frame_obs

    def _frame_world_transform_from_obs(self, obs_dict: Dict[str, torch.Tensor], frame_name: str) -> torch.Tensor:
        if frame_name == "base":
            pos = obs_dict["base_pos"][:, -1:]
            quat = obs_dict["base_quat"][:, -1:]
        elif frame_name == "left":
            if self._left_ref_pos_world_key in obs_dict:
                pos = obs_dict[self._left_ref_pos_world_key][:, -1:]
                quat = obs_dict[self._left_ref_quat_world_key][:, -1:]
            else:
                pos = obs_dict["left_ee_pos"][:, -1:]
                quat = obs_dict["left_ee_quat"][:, -1:]
        elif frame_name == "right":
            if self._right_ref_pos_world_key in obs_dict:
                pos = obs_dict[self._right_ref_pos_world_key][:, -1:]
                quat = obs_dict[self._right_ref_quat_world_key][:, -1:]
            else:
                pos = obs_dict["right_ee_pos"][:, -1:]
                quat = obs_dict["right_ee_quat"][:, -1:]
        else:
            raise ValueError(f"Unsupported frame '{frame_name}'.")
        return world_frame_to_transform(pos, quat, self.quat_to_mat)

    def _get_ee_refs(
        self, base_frame_obs_raw: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            "left_pos": base_frame_obs_raw["left_ee_pos"][:, -1:],
            "left_quat": base_frame_obs_raw["left_ee_quat"][:, -1:],
            "right_pos": base_frame_obs_raw["right_ee_pos"][:, -1:],
            "right_quat": base_frame_obs_raw["right_ee_quat"][:, -1:],
        }

    def _base_T_ee_frame(
        self, frame_name: str, base_frame_obs_raw: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # EE pose keys in base_frame_obs_raw are already expressed in the base
        # frame, so world_frame_to_transform here builds base_T_frame directly.
        if frame_name == "left":
            pos = base_frame_obs_raw["left_ee_pos"][:, -1:]
            quat = base_frame_obs_raw["left_ee_quat"][:, -1:]
        elif frame_name == "right":
            pos = base_frame_obs_raw["right_ee_pos"][:, -1:]
            quat = base_frame_obs_raw["right_ee_quat"][:, -1:]
        else:
            raise ValueError(f"Unsupported EE frame '{frame_name}'.")
        return world_frame_to_transform(pos, quat, self.quat_to_mat)

    def _base_pose_real_to_canonical_norm(
        self,
        base_pose_real: torch.Tensor,
        base_frame_obs_raw: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._canonical_space == "base":
            return self.normalizer["base_action"].normalize(base_pose_real)
        if self._canonical_space == "base_rel_trans":
            refs = self._get_ee_refs(base_frame_obs_raw)
            canonical_real = convert_base_pose_action_to_rel_trans(
                base_pose_real, refs["left_pos"], refs["right_pos"]
            )
            return self.normalizer["base_rel_trans_action"].normalize(canonical_real)
        if self._canonical_space in ("left", "right"):
            base_T_frame = self._base_T_ee_frame(
                self._canonical_space, base_frame_obs_raw
            ).to(dtype=base_pose_real.dtype)
            eye_T = eye_transform(
                base_pose_real.shape[0], 1,
                dtype=base_pose_real.dtype, device=base_pose_real.device,
            )
            canonical_real = convert_pose_action_between_frames_as_vectors(
                action=base_pose_real,
                world_T_source_frame=eye_T,
                world_T_target_frame=base_T_frame,
            )
            return self.normalizer[f"{self._canonical_space}_action"].normalize(canonical_real)
        refs = self._get_ee_refs(base_frame_obs_raw)
        canonical_real = convert_base_pose_action_to_rel_traj_as_vectors(
            base_pose_real,
            refs["left_pos"],
            refs["left_quat"],
            refs["right_pos"],
            refs["right_quat"],
            self.quat_to_mat,
        )
        return self.normalizer["rel_traj_action"].normalize(canonical_real)

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
            return convert_pose_action_between_frames_as_vectors(
                action=canonical_real,
                world_T_source_frame=base_T_frame,
                world_T_target_frame=eye_T,
            )
        refs = self._get_ee_refs(base_frame_obs_raw)
        if self._canonical_space == "base_rel_trans":
            canonical_real = self.normalizer["base_rel_trans_action"].unnormalize(canonical_sample_norm)
            return convert_base_rel_trans_to_pose_action(
                canonical_real, refs["left_pos"], refs["right_pos"]
            )
        canonical_real = self.normalizer["rel_traj_action"].unnormalize(canonical_sample_norm)
        return convert_rel_traj_to_base_pose_action_as_vectors(
            canonical_real,
            refs["left_pos"],
            refs["left_quat"],
            refs["right_pos"],
            refs["right_quat"],
            self.quat_to_mat,
        )

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
            return rotate_raw_vectors_between_frames(
                frame_noise_real,
                world_T_source_frame=base_T_frame,
                world_T_target_frame=eye_T,
            )
        refs = self._get_ee_refs(base_frame_obs_raw)
        rel_traj_noise_real = unnormalize_like_delta(canonical_noise_norm, self.normalizer["rel_traj_action"])
        return convert_rel_traj_delta_to_base(
            rel_traj_noise_real,
            refs["left_quat"],
            refs["right_quat"],
            self.quat_to_mat,
        )

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
            frame_noise_real = rotate_raw_vectors_between_frames(
                base_noise_real,
                world_T_source_frame=eye_T,
                world_T_target_frame=base_T_frame,
            )
            return normalize_like_delta(
                frame_noise_real, self.normalizer[f"{self._canonical_space}_action"]
            )
        refs = self._get_ee_refs(base_frame_obs_raw)
        rel_traj_noise_real = convert_base_delta_to_rel_traj(
            base_noise_real,
            refs["left_quat"],
            refs["right_quat"],
            self.quat_to_mat,
        )
        return normalize_like_delta(rel_traj_noise_real, self.normalizer["rel_traj_action"])

    def _build_router_train_metrics(
        self,
        router_probs: torch.Tensor,
        expert_names,
        timesteps: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        return self._build_router_weight_metrics(
            router_probs=router_probs,
            expert_names=expert_names,
            timesteps=timesteps,
            prefix="router",
        )

    def _build_router_weight_metrics(
        self,
        router_probs: torch.Tensor,
        expert_names,
        timesteps: torch.Tensor = None,
        prefix: str = "router",
    ) -> Dict[str, torch.Tensor]:
        reduce_dims = tuple(range(router_probs.ndim - 1))
        mean_probs = router_probs.mean(dim=reduce_dims)
        entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        max_prob = router_probs.max(dim=-1).values.mean()
        metrics = {
            f"{prefix}/entropy": entropy.detach(),
            f"{prefix}/max_prob": max_prob.detach(),
            f"{prefix}/mode_fixed_uniform": router_probs.new_tensor(
                1.0 if self.router_mode == "fixed_uniform" else 0.0
            ),
        }
        for idx, expert_name in enumerate(expert_names):
            metrics[f"{prefix}/weight_{expert_name}"] = mean_probs[idx].detach()
        if timesteps is not None:
            threshold = 0.5 * float(self.noise_scheduler.config.num_train_timesteps - 1)
            timestep_info = (
                ("t_low", timesteps.to(dtype=torch.float32) <= threshold),
                ("t_high", timesteps.to(dtype=torch.float32) > threshold),
            )
            for suffix, mask in timestep_info:
                if not mask.any():
                    continue
                probs_subset = router_probs[mask]
                subset_mean_probs = probs_subset.mean(dim=tuple(range(probs_subset.ndim - 1)))
                for expert_idx, expert_name in enumerate(expert_names):
                    metrics[f"{prefix}/weight_{expert_name}/{suffix}"] = subset_mean_probs[
                        expert_idx
                    ].detach()
        return metrics

    def get_last_train_metrics(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_train_metrics)

    @staticmethod
    def _masked_mean_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean()

    @staticmethod
    def _masked_batch_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean(dim=-1)

    def _compute_canonical_expert_losses(
        self,
        base_outputs: Dict[str, torch.Tensor],
        expert_names,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        losses = {}
        batch_losses = {}
        total = target.new_zeros(())
        for name in expert_names:
            batch_loss = self._masked_batch_mse(base_outputs[name], target, loss_mask)
            batch_losses[name] = batch_loss
            loss = batch_loss.mean()
            losses[name] = loss
            total = total + loss
        total = total / len(expert_names)
        return losses, batch_losses, total

    def _compute_native_expert_losses(
        self,
        model_out: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        loss_mask: torch.Tensor,
        batch_obs_repeated: Dict[str, torch.Tensor],
        frame_obs_raw_base_repeated: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Compute per-expert loss in each expert's native frame (epsilon prediction)."""
        losses = {}
        batch_losses = {}
        total = noise.new_zeros(())
        world_T_base = self._frame_world_transform_from_obs(batch_obs_repeated, "base")
        base_noise_real = self._canonical_noise_norm_to_base_delta_real(
            noise, frame_obs_raw_base_repeated
        )
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
                expert_noise_real = rotate_raw_vectors_between_frames(
                    base_noise_real, world_T_base, world_T_expert
                )
                expert_target = normalize_like_delta(expert_noise_real, self.normalizer[action_key])
            elif name in ("base", "base_rel_trans"):
                expert_target = normalize_like_delta(base_noise_real, self.normalizer[action_key])
            elif name == "rel_traj":
                refs = self._get_ee_refs(frame_obs_raw_base_repeated)
                expert_noise_real = convert_base_delta_to_rel_traj(
                    base_noise_real,
                    refs["left_quat"].to(base_noise_real.dtype),
                    refs["right_quat"].to(base_noise_real.dtype),
                    self.quat_to_mat,
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

    def _record_expert_loss_metrics(
        self,
        canonical_losses: Dict[str, torch.Tensor],
        canonical_total: torch.Tensor,
        native_losses: Dict[str, torch.Tensor] = None,
        native_total: torch.Tensor = None,
    ) -> None:
        for name, loss in canonical_losses.items():
            self._last_train_metrics[f"expert_loss_canonical/{name}"] = loss.detach()
        self._last_train_metrics["expert_loss_canonical/total"] = canonical_total.detach()
        if native_losses is not None and native_total is not None:
            for name, loss in native_losses.items():
                self._last_train_metrics[f"expert_loss_native/{name}"] = loss.detach()
            self._last_train_metrics["expert_loss_native/total"] = native_total.detach()

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

        # Convert canonical sample to base pose real (all experts derive from this).
        # Use vector-preserving path to avoid Gram-Schmidt on noisy rot6d.
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
                left_sample_real = convert_pose_action_between_frames_as_vectors(
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
                right_sample_real = convert_pose_action_between_frames_as_vectors(
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
                # base_pose_real may be noisy x_t — use as_vectors to avoid
                # Gram-Schmidt on invalid rot6d columns.
                rel_traj_real = convert_base_pose_action_to_rel_traj_as_vectors(
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

    def _convert_expert_output_to_canonical_norm(
        self,
        expert_output_norm: torch.Tensor,
        expert_name: str,
        obs_dict: Dict[str, torch.Tensor],
        base_frame_obs_raw: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert expert epsilon prediction to canonical normalized space."""
        world_T_base = self._frame_world_transform_from_obs(obs_dict, "base")
        # All experts convert to base-frame noise real, then normalize to canonical
        if expert_name == "base":
            base_noise_real = unnormalize_like_delta(expert_output_norm, self.normalizer["base_action"])
        elif expert_name in ("left", "right"):
            world_T_frame = self._frame_world_transform_from_obs(obs_dict, expert_name)
            frame_real = unnormalize_like_delta(expert_output_norm, self.normalizer[f"{expert_name}_action"])
            base_noise_real = rotate_raw_vectors_between_frames(frame_real, world_T_frame, world_T_base)
        elif expert_name == "base_rel_trans":
            base_noise_real = unnormalize_like_delta(expert_output_norm, self.normalizer["base_rel_trans_action"])
        elif expert_name == "rel_traj":
            refs = self._get_ee_refs(base_frame_obs_raw)
            rel_traj_noise_real = unnormalize_like_delta(
                expert_output_norm, self.normalizer["rel_traj_action"]
            )
            base_noise_real = convert_rel_traj_delta_to_base(
                rel_traj_noise_real,
                refs["left_quat"],
                refs["right_quat"],
                self.quat_to_mat,
            )
        else:
            raise ValueError(f"Unsupported expert '{expert_name}'.")
        return self._base_noise_real_to_canonical_norm(base_noise_real, base_frame_obs_raw)

    def _mix_base_outputs(self, expert_outputs: Dict[str, torch.Tensor], router_probs: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([expert_outputs[name] for name in self.model.enabled_experts], dim=1)
        return torch.sum(stacked * router_probs[:, :, None, None], dim=1)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        frame_obs_norm,
        rgb_features,
        obs_dict,
        base_frame_obs_raw,
        generator=None,
        **kwargs,
    ):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            expert_inputs = self._build_expert_inputs(
                canonical_sample_norm=trajectory,
                frame_obs_norm=frame_obs_norm,
                rgb_features=rgb_features,
                obs_dict=obs_dict,
                base_frame_obs_raw=base_frame_obs_raw,
            )
            model_out = self.model(expert_inputs=expert_inputs, timestep=t, return_aux=True)
            base_outputs = {
                name: self._convert_expert_output_to_canonical_norm(
                    expert_output_norm=model_out["expert_outputs"][name],
                    expert_name=name,
                    obs_dict=obs_dict,
                    base_frame_obs_raw=base_frame_obs_raw,
                )
                for name in model_out["expert_names"]
            }
            mixed = self._mix_base_outputs(base_outputs, model_out["router_probs"])
            trajectory = self.noise_scheduler.step(
                mixed, t, trajectory, generator=generator, **kwargs
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _clamp_world_action(self, abs_action: torch.Tensor) -> torch.Tensor:
        action_min = (
            self.normalizer["action"]
            .params_dict.input_stats["min"]
            .to(device=abs_action.device, dtype=abs_action.dtype)
            .view(1, 1, -1)
        )
        action_max = (
            self.normalizer["action"]
            .params_dict.input_stats["max"]
            .to(device=abs_action.device, dtype=abs_action.dtype)
            .view(1, 1, -1)
        )
        return torch.clamp(abs_action, min=action_min, max=action_max)

    def _is_batch_obs(self, obs_dict: Dict[str, torch.Tensor]) -> bool:
        return any(key.startswith("base__") for key in obs_dict)

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
            base_action = convert_rel_traj_to_base_pose_action_as_vectors(
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
            base_action = convert_pose_action_between_frames_as_vectors(
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
        # Convert rot6d from internal column convention back to pytorch3d row
        # convention expected by the external world (replay buffer, evaluation).
        abs_action = action_rot6d_column_to_row(abs_action)
        abs_action = self._clamp_world_action(abs_action)
        start = To - 1
        end = start + self.n_action_steps
        return {"action": abs_action[:, start:end], "action_pred": abs_action}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if hasattr(self.obs_encoder, "set_normalizer"):
            self.obs_encoder.set_normalizer(self.normalizer)

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        encoder_backbone_lr: float,
        encoder_backbone_weight_decay: float = None,
        betas: Tuple[float, float] = (0.95, 0.999),
        **kwargs,
    ) -> torch.optim.Optimizer:
        if encoder_backbone_weight_decay is None:
            encoder_backbone_weight_decay = weight_decay
        backbone_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("obs_encoder.key_model_map"):
                backbone_params.append(param)
            else:
                other_params.append(param)
        optim_groups = [{"params": other_params, "weight_decay": weight_decay}]
        if backbone_params:
            optim_groups.append(
                {
                    "params": backbone_params,
                    "weight_decay": encoder_backbone_weight_decay,
                    "lr": encoder_backbone_lr,
                }
            )
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **kwargs)

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        batch = copy.deepcopy(batch)

        frame_obs_raw = self._prepare_batch_frame_obs(batch["obs"])
        frame_obs_norm = {
            frame_name: self._normalize_frame_obs(frame_obs_raw[frame_name], frame_name)
            for frame_name in self.FRAME_NAMES
        }
        trajectory = self.normalizer[self._train_action_key].normalize(batch[self._train_action_key])
        rgb_features = self.obs_encoder.forward_rgb(frame_obs_norm["base"])

        frame_obs_norm = {
            frame_name: repeat_train_diffusion_dict(frame_obs_norm[frame_name], self.train_diffusion_n_samples)
            for frame_name in self.FRAME_NAMES
        }
        rgb_features = {
            key: repeat_train_diffusion_tensor(val, self.train_diffusion_n_samples)
            for key, val in rgb_features.items()
        }
        trajectory = repeat_train_diffusion_tensor(trajectory, self.train_diffusion_n_samples)
        batch_obs_repeated = repeat_train_diffusion_dict(batch["obs"], self.train_diffusion_n_samples)
        frame_obs_raw_base_repeated = repeat_train_diffusion_dict(
            frame_obs_raw["base"], self.train_diffusion_n_samples
        )
        cond_data = trajectory
        bsz = trajectory.shape[0]
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        expert_inputs = self._build_expert_inputs(
            canonical_sample_norm=noisy_trajectory,
            frame_obs_norm=frame_obs_norm,
            rgb_features=rgb_features,
            obs_dict=batch_obs_repeated,
            base_frame_obs_raw=frame_obs_raw_base_repeated,
        )
        model_out = self.model(expert_inputs=expert_inputs, timestep=timesteps, return_aux=True)
        base_outputs = {
            name: self._convert_expert_output_to_canonical_norm(
                expert_output_norm=model_out["expert_outputs"][name],
                expert_name=name,
                obs_dict=batch_obs_repeated,
                base_frame_obs_raw=frame_obs_raw_base_repeated,
            )
            for name in model_out["expert_names"]
        }
        router_probs = model_out["router_probs"]
        pred = self._mix_base_outputs(base_outputs, router_probs)

        loss = self._masked_mean_mse(pred, noise, loss_mask)

        self._last_train_metrics = self._build_router_train_metrics(
            router_probs=router_probs,
            expert_names=model_out["expert_names"],
            timesteps=timesteps,
        )
        (
            canonical_losses,
            _canonical_batch_losses,
            expert_canonical_loss_total,
        ) = self._compute_canonical_expert_losses(
            base_outputs=base_outputs,
            expert_names=model_out["expert_names"],
            target=noise,
            loss_mask=loss_mask,
        )

        native_losses = None
        expert_loss_native_total = None
        if self.expert_loss_coef > 0.0:
            (
                native_losses,
                _native_batch_losses,
                expert_loss_native_total,
            ) = self._compute_native_expert_losses(
                model_out=model_out,
                noise=noise,
                loss_mask=loss_mask,
                batch_obs_repeated=batch_obs_repeated,
                frame_obs_raw_base_repeated=frame_obs_raw_base_repeated,
            )
            loss = loss + self.expert_loss_coef * expert_loss_native_total
        self._record_expert_loss_metrics(
            canonical_losses=canonical_losses,
            canonical_total=expert_canonical_loss_total,
            native_losses=native_losses,
            native_total=expert_loss_native_total,
        )
        return loss
