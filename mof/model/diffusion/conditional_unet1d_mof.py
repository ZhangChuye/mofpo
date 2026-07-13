from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from mof.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mof.model.diffusion.positional_embedding import SinusoidalPosEmb


class ConditionalUnet1DMoF(nn.Module):
    _ALLOWED_EXPERTS = ("base", "left", "right", "base_rel_trans", "rel_traj")
    _ALLOWED_ROUTER_MODES = ("learned", "fixed_uniform")

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        low_dim_cond_dim: int,
        horizon: int,
        local_cond_dim=None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        N: int = 8,
        rgb_keys=None,
        low_dim_keys=None,
        enabled_experts: Sequence[str] = ("base", "left", "right"),
        router_mode: str = "learned",
        router_hidden_dim: int = 256,
        router_timestep_embed_dim: int = 128,
    ):
        super().__init__()
        if local_cond_dim is not None:
            raise ValueError("ConditionalUnet1DMoF currently supports local_cond_dim=None only.")

        enabled_experts = tuple(enabled_experts)
        invalid_experts = [name for name in enabled_experts if name not in self._ALLOWED_EXPERTS]
        if invalid_experts:
            raise ValueError(
                f"Unsupported expert names {invalid_experts}. Expected subset of {self._ALLOWED_EXPERTS}."
            )
        if not enabled_experts:
            raise ValueError("ConditionalUnet1DMoF requires at least one enabled expert.")
        if router_mode not in self._ALLOWED_ROUTER_MODES:
            raise ValueError(
                f"Unsupported router_mode '{router_mode}'. Expected one of {self._ALLOWED_ROUTER_MODES}."
            )
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.low_dim_cond_dim = low_dim_cond_dim
        self.horizon = horizon
        self.N = N
        self.rgb_keys = list(rgb_keys or [])
        self.low_dim_keys = list(low_dim_keys or [])
        self.enabled_experts = enabled_experts
        self.router_mode = router_mode

        expert_kwargs = dict(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
            rgb_keys=rgb_keys,
            low_dim_keys=low_dim_keys,
        )
        experts = nn.ModuleDict()
        for expert_name in enabled_experts:
            experts[expert_name] = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )
        self.experts = experts

        self.router_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(router_timestep_embed_dim),
            nn.Linear(router_timestep_embed_dim, router_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(router_timestep_embed_dim * 4, router_timestep_embed_dim),
        )
        self.rgb_flat_dim = self.global_cond_dim - self.low_dim_cond_dim
        # Router input: shared RGB + per-expert low-dim (dedup).
        self._has_base_conditioned_expert = any(
            expert_name in self.enabled_experts
            for expert_name in ("base", "base_rel_trans", "rel_traj")
        )
        router_input_dim = self.rgb_flat_dim
        if self._has_base_conditioned_expert:
            router_input_dim += self.low_dim_cond_dim
        if "left" in self.enabled_experts:
            router_input_dim += self.low_dim_cond_dim
        if "right" in self.enabled_experts:
            router_input_dim += self.low_dim_cond_dim
        router_output_dim = len(self.enabled_experts)
        if self.router_mode == "learned":
            self.router = nn.Sequential(
                nn.Linear(router_input_dim + router_timestep_embed_dim, router_hidden_dim),
                nn.Mish(),
                nn.Linear(router_hidden_dim, router_hidden_dim),
                nn.Mish(),
                nn.Linear(router_hidden_dim, router_output_dim),
            )
        else:
            self.router = None

    def _prepare_timesteps(self, timestep, batch_size: int, device) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(device=device)
        elif timesteps.dim() != 1:
            raise ValueError(f"Expected scalar or 1D timestep tensor, got shape {timesteps.shape}")
        return timesteps.expand(batch_size).to(device=device, dtype=torch.long)

    def _flatten_rgb_features(self, rgb_features: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        flat_features: List[torch.Tensor] = []
        for key in sorted(rgb_features.keys()):
            feat = rgb_features[key]
            if feat.ndim == 3:
                feat = feat.reshape(batch_size, -1)
            flat_features.append(feat)
        if not flat_features:
            raise ValueError("ConditionalUnet1DMoF requires non-empty rgb_features.")
        return torch.cat(flat_features, dim=-1)

    def _flatten_low_dim_obs(self, obs_dict: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        flat_obs = []
        for key in self.low_dim_keys:
            if key not in obs_dict:
                raise KeyError(f"Missing low-dim conditioning key '{key}' for Mixture of Frames UNet.")
            flat_obs.append(obs_dict[key].reshape(batch_size, -1))
        if not flat_obs:
            example = next(iter(obs_dict.values()))
            return torch.zeros(batch_size, 0, device=example.device, dtype=example.dtype)
        return torch.cat(flat_obs, dim=-1)

    def _get_shared_rgb_flat(self, expert_inputs: Dict[str, Dict[str, torch.Tensor]], batch_size: int) -> torch.Tensor:
        first_expert_name = self.enabled_experts[0]
        return self._flatten_rgb_features(
            expert_inputs[first_expert_name]["rgb_features"], batch_size=batch_size
        )

    def _get_base_low_dim_flat(
        self, expert_inputs: Dict[str, Dict[str, torch.Tensor]], batch_size: int
    ) -> torch.Tensor:
        for expert_name in ("base", "base_rel_trans", "rel_traj"):
            if expert_name in expert_inputs:
                return self._flatten_low_dim_obs(
                    expert_inputs[expert_name]["obs_dict"], batch_size=batch_size
                )
        raise ValueError("No base-conditioned expert available for router base_low_dim input.")

    def _compute_router_probs(self, expert_inputs: Dict[str, Dict[str, torch.Tensor]], timestep, batch_size: int, action_horizon: int):
        num_experts = len(self.enabled_experts)
        if self.router_mode == "fixed_uniform":
            device = next(iter(expert_inputs.values()))["sample"].device
            dtype = next(iter(expert_inputs.values()))["sample"].dtype
            output_shape = [batch_size, num_experts]
            probs = torch.full(
                output_shape,
                1.0 / num_experts,
                device=device,
                dtype=dtype,
            )
            logits = torch.zeros_like(probs)
            return probs, logits
        sample = next(iter(expert_inputs.values()))["sample"]
        device = sample.device
        dtype = sample.dtype
        # Shared RGB + per-expert low-dim (dedup).
        router_inputs = [self._get_shared_rgb_flat(expert_inputs, batch_size=batch_size)]
        if self._has_base_conditioned_expert:
            router_inputs.append(
                self._get_base_low_dim_flat(expert_inputs, batch_size=batch_size)
            )
        if "left" in self.enabled_experts:
            router_inputs.append(
                self._flatten_low_dim_obs(expert_inputs["left"]["obs_dict"], batch_size=batch_size)
            )
        if "right" in self.enabled_experts:
            router_inputs.append(
                self._flatten_low_dim_obs(expert_inputs["right"]["obs_dict"], batch_size=batch_size)
            )
        router_input = torch.cat(router_inputs, dim=-1)
        timestep_tensor = self._prepare_timesteps(timestep=timestep, batch_size=batch_size, device=device)
        timestep_embed = self.router_timestep_encoder(timestep_tensor.to(dtype=dtype))
        logits = self.router(torch.cat([router_input, timestep_embed], dim=-1))

        logits = logits.reshape(batch_size, num_experts)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits

    def forward(self, expert_inputs: Dict[str, Dict[str, torch.Tensor]], timestep, return_aux: bool = False):
        expert_outputs = {}
        batch_size = None
        action_horizon = None
        for expert_name in self.enabled_experts:
            expert_input = expert_inputs[expert_name]
            sample = expert_input["sample"]
            if batch_size is None:
                batch_size = sample.shape[0]
                action_horizon = sample.shape[1]
            global_cond = torch.cat(
                [
                    self._flatten_rgb_features(expert_input["rgb_features"], batch_size=batch_size),
                    self._flatten_low_dim_obs(expert_input["obs_dict"], batch_size=batch_size),
                ],
                dim=-1,
            )
            out = self.experts[expert_name](
                sample=sample,
                timestep=timestep,
                local_cond=None,
                global_cond=global_cond,
            )
            expert_outputs[expert_name] = out

        router_probs, router_logits = self._compute_router_probs(
            expert_inputs=expert_inputs,
            timestep=timestep,
            batch_size=batch_size,
            action_horizon=action_horizon,
        )
        if not return_aux:
            return expert_outputs
        return {
            "expert_outputs": expert_outputs,
            "router_probs": router_probs,
            "router_logits": router_logits,
            "expert_names": self.enabled_experts,
        }
