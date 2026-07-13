"""Ensemble baseline policy for the MoF comparison.

Controlled ablation against ``router_mode=fixed_uniform`` Mixture of Frames: N experts
all operate in the *same* canonical frame (no cross-frame routing). Outputs are
averaged uniformly.  Any gain over a single-expert baseline is attributable to
ensembling alone, not to frame mixing.
"""
from typing import Dict, Tuple, Union
import copy
import inspect

import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mof.common.mof_transform_util import (
    action_rot6d_column_to_row,
    convert_base_rel_trans_to_pose_action,
    convert_frame_action_to_world,
    convert_rel_traj_to_base_pose_action_as_vectors,
    transform_obs_pose_dict_to_frame,
    world_frame_to_transform,
)
from mof.model.common.normalizer import LinearNormalizer
from mof.model.common.rotation_transformer import RotationTransformer
from mof.model.diffusion.conditional_unet1d_ensemble import ConditionalUnet1DEnsemble
from mof.model.diffusion.mask_generator import LowdimMaskGenerator
from mof.model.vision.timm_obs_encoder import TimmObsEncoder
from mof.policy.base_image_policy import BaseImagePolicy
from mof.policy.train_diffusion_utils import (
    repeat_train_diffusion_tensor,
)


class DiffusionUnetPolicyEnsemble(BaseImagePolicy):
    """Uniform ensemble of N independent denoisers in a single canonical frame.

    Reuses the MoF dataset so that direct A/B comparison against the
    ``fixed_uniform`` MoE ablation is possible (identical action / obs streams).
    Only the *base* frame obs are used as conditioning; ``left``/``right`` frame
    obs from the dataset are ignored.
    """

    # Observations still arrive with base/left/right frame prefixes because the
    # ensemble reuses the MoF dataset, but only the base frame is read.
    _COND_FRAME = "base"

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
        num_experts: int = 3,
        canonical_space: str = "base",
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond, "Ensemble policy currently supports obs_as_global_cond=True only."
        if canonical_space not in ("base", "base_rel_trans", "rel_traj"):
            raise ValueError(
                f"Unsupported canonical_space '{canonical_space}'. "
                f"Expected 'base', 'base_rel_trans', or 'rel_traj'."
            )
        if int(num_experts) < 1:
            raise ValueError(f"num_experts must be >=1, got {num_experts}.")

        # World-frame reference keys the dataset supplies for reconstruction.
        self._left_ref_pos_world_key = "_left_ref_pos_world"
        self._left_ref_quat_world_key = "_left_ref_quat_world"
        self._right_ref_pos_world_key = "_right_ref_pos_world"
        self._right_ref_quat_world_key = "_right_ref_quat_world"

        # base_pos/base_quat are inputs to the world-frame reconstruction only,
        # not conditioning for the denoiser — drop them from encoder inputs.
        if hasattr(obs_encoder, "low_dim_keys"):
            obs_encoder.low_dim_keys = [
                key for key in obs_encoder.low_dim_keys if key not in ("base_pos", "base_quat")
            ]

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # Relies on TimmObsEncoder.output_shape() recomputing from the live
        # self.low_dim_keys list (not a cached value) so the mutation above
        # is honored here.
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.obs_encoder = obs_encoder
        self.shape_meta = shape_meta
        self.num_experts = int(num_experts)
        self.model = ConditionalUnet1DEnsemble(
            num_experts=self.num_experts,
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
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
        elif canonical_space == "base_rel_trans":
            self._train_action_key = "base_rel_trans_action"
        else:
            self._train_action_key = "rel_traj_action"
        self._last_train_metrics: Dict[str, torch.Tensor] = dict()

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

    # ---------- obs handling ----------

    def _frame_obs_key(self, frame_name: str, key: str) -> str:
        return f"{frame_name}__{key}"

    def _normalize_cond_frame_obs(self, cond_frame_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, value in cond_frame_obs.items():
            if key in getattr(self.obs_encoder, "rgb_keys", []):
                out[key] = self.normalizer[key].normalize(value)
            else:
                out[key] = self.normalizer[self._frame_obs_key(self._COND_FRAME, key)].normalize(value)
        return out

    def _extract_batch_cond_frame_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        frame_obs = {}
        for key in getattr(self.obs_encoder, "rgb_keys", []):
            frame_obs[key] = obs_dict[key]
        for key in getattr(self.obs_encoder, "low_dim_keys", []):
            frame_obs[key] = obs_dict[self._frame_obs_key(self._COND_FRAME, key)]
        return frame_obs

    def _prepare_runtime_cond_frame_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        world_obs = {
            key: value
            for key, value in obs_dict.items()
            if key in getattr(self.obs_encoder, "low_dim_keys", [])
        }
        base_T = world_frame_to_transform(obs_dict["base_pos"], obs_dict["base_quat"], self.quat_to_mat)
        frame_obs = transform_obs_pose_dict_to_frame(world_obs, base_T, self.quat_to_mat, self.mat_to_quat)
        for key in getattr(self.obs_encoder, "rgb_keys", []):
            frame_obs[key] = obs_dict[key]
        return frame_obs

    def _base_world_transform_from_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        pos = obs_dict["base_pos"][:, -1:]
        quat = obs_dict["base_quat"][:, -1:]
        return world_frame_to_transform(pos, quat, self.quat_to_mat)

    def _get_ee_refs_from_batch(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fetch EE references in the *base* frame — needed for canonical-space
        reconstruction in rel_trans / rel_traj modes."""
        left_pos = obs_dict[self._frame_obs_key(self._COND_FRAME, "left_ee_pos")][:, -1:]
        left_quat = obs_dict[self._frame_obs_key(self._COND_FRAME, "left_ee_quat")][:, -1:]
        right_pos = obs_dict[self._frame_obs_key(self._COND_FRAME, "right_ee_pos")][:, -1:]
        right_quat = obs_dict[self._frame_obs_key(self._COND_FRAME, "right_ee_quat")][:, -1:]
        return {
            "left_pos": left_pos,
            "left_quat": left_quat,
            "right_pos": right_pos,
            "right_quat": right_quat,
        }

    def _get_ee_refs_runtime(self, cond_frame_obs_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "left_pos": cond_frame_obs_raw["left_ee_pos"][:, -1:],
            "left_quat": cond_frame_obs_raw["left_ee_quat"][:, -1:],
            "right_pos": cond_frame_obs_raw["right_ee_pos"][:, -1:],
            "right_quat": cond_frame_obs_raw["right_ee_quat"][:, -1:],
        }

    # ---------- metric helpers ----------

    @staticmethod
    def _masked_mean_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean()

    def _compute_per_expert_mse(
        self,
        per_expert_preds,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return per-expert MSE for *logging only*.

        Since all experts share identical inputs (same obs, same noisy sample,
        same timestep) and target, this does not measure diversity — it's a
        cheap monitoring signal for spotting an individual expert diverging.
        Do not use as an auxiliary loss term: it wouldn't regularize anything
        beyond the averaged MSE.
        """
        return {
            f"expert_{idx}": self._masked_mean_mse(pred, target, loss_mask)
            for idx, pred in enumerate(per_expert_preds)
        }

    def get_last_train_metrics(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_train_metrics)

    # ---------- canonical <-> world reconstruction ----------

    def _canonical_sample_norm_to_world_action(
        self,
        canonical_sample_norm: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
        cond_frame_obs_raw: Dict[str, torch.Tensor] = None,
    ) -> torch.Tensor:
        canonical_action = self.normalizer[self._train_action_key].unnormalize(canonical_sample_norm)
        if self._canonical_space == "base_rel_trans":
            refs = self._get_ee_refs_runtime(cond_frame_obs_raw) if cond_frame_obs_raw is not None else self._get_ee_refs_from_batch(obs_dict)
            base_action = convert_base_rel_trans_to_pose_action(
                canonical_action,
                refs["left_pos"].to(canonical_action.dtype),
                refs["right_pos"].to(canonical_action.dtype),
            )
        elif self._canonical_space == "rel_traj":
            refs = self._get_ee_refs_runtime(cond_frame_obs_raw) if cond_frame_obs_raw is not None else self._get_ee_refs_from_batch(obs_dict)
            base_action = convert_rel_traj_to_base_pose_action_as_vectors(
                canonical_action,
                refs["left_pos"].to(canonical_action.dtype),
                refs["left_quat"].to(canonical_action.dtype),
                refs["right_pos"].to(canonical_action.dtype),
                refs["right_quat"].to(canonical_action.dtype),
                self.quat_to_mat,
            )
        else:
            base_action = canonical_action
        abs_action = convert_frame_action_to_world(
            action_frame=base_action,
            world_T_frame=self._base_world_transform_from_obs(obs_dict),
        )
        abs_action = action_rot6d_column_to_row(abs_action)
        return self._clamp_world_action(abs_action)

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
        return any(key.startswith(f"{self._COND_FRAME}__") for key in obs_dict)

    # ---------- inference ----------

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        global_cond,
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
            model_out = self.model(
                sample=trajectory,
                timestep=t,
                global_cond=global_cond,
                local_cond=None,
            )
            trajectory = self.noise_scheduler.step(
                model_out, t, trajectory, generator=generator, **kwargs
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        if self._is_batch_obs(obs_dict):
            cond_frame_obs_raw = self._extract_batch_cond_frame_obs(obs_dict)
            cond_refs_source = None
        else:
            cond_frame_obs_raw = self._prepare_runtime_cond_frame_obs(obs_dict)
            cond_refs_source = cond_frame_obs_raw
        cond_frame_obs_norm = self._normalize_cond_frame_obs(cond_frame_obs_raw)

        value = next(iter(cond_frame_obs_norm.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        global_cond = self.obs_encoder(cond_frame_obs_norm)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            **self.kwargs,
        )
        abs_action = self._canonical_sample_norm_to_world_action(
            canonical_sample_norm=nsample,
            obs_dict=obs_dict,
            cond_frame_obs_raw=cond_refs_source,
        )
        start = To - 1
        end = start + self.n_action_steps
        return {"action": abs_action[:, start:end], "action_pred": abs_action}

    # ---------- training ----------

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

        cond_frame_obs_raw = self._extract_batch_cond_frame_obs(batch["obs"])
        cond_frame_obs_norm = self._normalize_cond_frame_obs(cond_frame_obs_raw)
        trajectory = self.normalizer[self._train_action_key].normalize(batch[self._train_action_key])

        # Run the RGB backbone once on the unrepeated batch, then repeat the
        # flat conditioning vector — avoids an N× backbone cost when
        # train_diffusion_n_samples > 1.
        global_cond = self.obs_encoder(cond_frame_obs_norm)
        global_cond = repeat_train_diffusion_tensor(global_cond, self.train_diffusion_n_samples)
        trajectory = repeat_train_diffusion_tensor(trajectory, self.train_diffusion_n_samples)

        bsz = trajectory.shape[0]

        cond_data = trajectory
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

        model_out = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=global_cond,
            local_cond=None,
            return_per_expert=True,
        )
        pred = model_out["mean"]
        loss = self._masked_mean_mse(pred, noise, loss_mask)

        per_expert_mse = self._compute_per_expert_mse(
            per_expert_preds=model_out["per_expert"],
            target=noise,
            loss_mask=loss_mask,
        )
        self._last_train_metrics = {
            "ensemble/num_experts": torch.tensor(float(self.num_experts)),
            "ensemble/loss_mean": loss.detach(),
        }
        for name, expert_mse in per_expert_mse.items():
            self._last_train_metrics[f"ensemble/loss_{name}"] = expert_mse.detach()
        return loss
