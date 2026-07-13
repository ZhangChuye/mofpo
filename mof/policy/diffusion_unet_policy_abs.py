from typing import Dict, Union
import copy

import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from mof.common.pytorch_util import dict_apply
from mof.model.common.normalizer import LinearNormalizer
from mof.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mof.model.diffusion.mask_generator import LowdimMaskGenerator
from mof.model.vision.timm_obs_encoder import TimmObsEncoder
from mof.policy.base_image_policy import BaseImagePolicy
from mof.policy.train_diffusion_utils import (
    repeat_train_diffusion_tensor,
)


class DiffusionUnetPolicyAbs(BaseImagePolicy):
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
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond, "This UNet wrapper currently supports obs_as_global_cond=True only."

        self._recon_only_obs_keys = ("base_pos", "base_quat")
        if hasattr(obs_encoder, "low_dim_keys"):
            obs_encoder.low_dim_keys = [
                key for key in obs_encoder.low_dim_keys if key not in self._recon_only_obs_keys
            ]

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
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
        self.kwargs = kwargs

        self._train_action_key = "action"
        self._pred_action_norm_key = "action"

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _get_cond_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in obs_dict.items()
            if key not in self._recon_only_obs_keys
        }

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        global_cond,
        nobs,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                sample=trajectory,
                timestep=t,
                global_cond=global_cond,
            )
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(self._get_cond_obs(obs_dict))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        global_cond = self.obs_encoder.forward(nobs)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            nobs=nobs,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer[self._pred_action_norm_key].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if hasattr(self.obs_encoder, "set_normalizer"):
            self.obs_encoder.set_normalizer(self.normalizer)

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        batch = copy.deepcopy(batch)

        nobs = self.normalizer.normalize(self._get_cond_obs(batch["obs"]))
        nactions = self.normalizer[self._train_action_key].normalize(batch[self._train_action_key])
        trajectory = nactions
        cond_data = trajectory
        global_cond = self.obs_encoder.forward(nobs)
        global_cond = repeat_train_diffusion_tensor(
            global_cond, self.train_diffusion_n_samples
        )
        trajectory = repeat_train_diffusion_tensor(
            trajectory, self.train_diffusion_n_samples
        )
        cond_data = repeat_train_diffusion_tensor(
            cond_data, self.train_diffusion_n_samples
        )
        bsz = trajectory.shape[0]
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
