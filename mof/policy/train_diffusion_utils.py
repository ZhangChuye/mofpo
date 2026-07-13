from typing import Dict, Optional, Union

import torch

from mof.common.pytorch_util import dict_apply


TrainDiffusionValue = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]


def repeat_train_diffusion_tensor(
    x: TrainDiffusionValue,
    repeats: int,
) -> TrainDiffusionValue:
    if x is None or repeats == 1:
        return x
    if isinstance(x, dict):
        return repeat_train_diffusion_dict(x, repeats)
    return torch.repeat_interleave(x, repeats=repeats, dim=0)


def repeat_train_diffusion_dict(
    obs_dict: Optional[Dict[str, torch.Tensor]],
    repeats: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if obs_dict is None or repeats == 1:
        return obs_dict
    return dict_apply(
        obs_dict,
        lambda x: torch.repeat_interleave(x, repeats=repeats, dim=0),
    )
