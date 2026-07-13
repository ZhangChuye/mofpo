"""Ensemble baseline for the MoF comparison.

Wraps N independently-initialized :class:`ConditionalUnet1D` experts that all
operate in the same canonical frame.  Outputs are averaged (uniform ensemble).
Used to test whether the benefit of ``router_mode=fixed_uniform`` in Mixture of Frames
comes from frame diversity or purely from ensembling multiple models.
"""
from typing import List

import torch
import torch.nn as nn

from mof.model.diffusion.conditional_unet1d import ConditionalUnet1D


class ConditionalUnet1DEnsemble(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        global_cond_dim: int,
        local_cond_dim=None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        if local_cond_dim is not None:
            raise ValueError(
                "ConditionalUnet1DEnsemble currently supports local_cond_dim=None only."
            )
        if int(num_experts) < 1:
            raise ValueError(f"num_experts must be >=1, got {num_experts}.")
        self.num_experts = int(num_experts)
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim

        self.experts = nn.ModuleList(
            [
                ConditionalUnet1D(
                    input_dim=input_dim,
                    local_cond_dim=None,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: torch.Tensor,
        local_cond=None,
        return_per_expert: bool = False,
    ):
        """Run all N experts and return the averaged epsilon prediction.

        When ``return_per_expert`` is True, also returns the list of individual
        expert predictions (useful for per-expert auxiliary loss).
        """
        if local_cond is not None:
            raise ValueError("local_cond is not supported in ensemble variant.")
        per_expert: List[torch.Tensor] = [
            expert(
                sample=sample,
                timestep=timestep,
                local_cond=None,
                global_cond=global_cond,
            )
            for expert in self.experts
        ]
        mean = torch.stack(per_expert, dim=0).mean(dim=0)
        if return_per_expert:
            return {"mean": mean, "per_expert": per_expert}
        return mean
