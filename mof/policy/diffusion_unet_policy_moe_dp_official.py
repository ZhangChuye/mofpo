"""Faithful port of the official MoE-DP policy (AlanxChen/moedp,
moe-dp/moe_dp/policy/dp_unet_mlp_moe.py) into our codebase.

Only the minimal edits needed to run in our environment:
  1. Imports rewritten to our package paths.
  2. New `pretrain` knob (default True). When True, the robomimic
     from-scratch ResNet18Conv inside the obs_encoder is swapped for a
     Timm-pretrained ResNet18 wrapper that exposes the same interface
     (`output_shape` and an `nn.Sequential` named `nets`). The swap happens
     BEFORE the official BN->GN sweep, so the GN replacement catches both
     robomimic-built modules and the Timm backbone uniformly. When False,
     the encoder is built exactly as the official release does (robomimic
     from-scratch ResNet18 with random conv init, then optional BN->GN).
  3. The `_obs_in_base_frame` flag tensor injected by our dex / bigym
     datasets (used by base-frame pipelines but not a real observation) is
     dropped from `shape_meta` and from `obs_dict` at runtime.
  4. `compute_loss` returns just the scalar loss (vs. official's
     `(loss, loss_dict)`) so it plugs into our existing workspace; the dict
     is stashed and exposed via `get_last_train_metrics()` for logging.
  5. New `train_action_key` knob (default "action"). When "action", the
     policy predicts world-frame absolute actions exactly like the official
     release. When "rel_action", it predicts gripper-relative trajectory
     deltas: at training the dataset's `rel_action` target is used; at
     inference, the obs are first transformed to base frame
     (`_get_base_frame_obs`) and the predicted rel_action is reconstructed
     to a world-frame absolute action (`_reconstruct_abs_action_from_rel_trans`)
     before returning to the env. In rel_action mode, `base_pos` / `base_quat`
     are filtered from the encoder input (kept around in `obs_dict` for the
     world-frame reconstruction) — exactly the convention used by our
     existing `DiffusionUnetPolicyMoEDPRelTraj`.

Everything else — per-obs-step routing, n_emb=256 bottleneck, 4*n_emb
expert hidden, top-k weight renormalization, 2-layer MLP gate with dropout,
Xavier init, ReLU activation, DDPMScheduler, [512,1024,2048] UNet — is
inherited verbatim from the official code.
"""
import math
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import reduce
from termcolor import cprint

# Diffuser scheduler — official uses DDPM, we follow.
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Our package paths (replacing official's `moe_dp.*`).
from mof.common.pytorch_util import dict_apply, replace_submodules
from mof.common.robomimic_config_util import get_robomimic_config
from mof.model.common.normalizer import LinearNormalizer
from mof.model.common.rotation_transformer import RotationTransformer
from mof.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mof.model.diffusion.mask_generator import LowdimMaskGenerator
from mof.model.moe.mlp_moe_official import MoE
from mof.policy.base_image_policy import BaseImagePolicy

# robomimic imports (matching the official policy line-for-line).
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.models.base_nets as rmbn_base
try:
    if not hasattr(rmbn_base, "CropRandomizer"):
        raise ImportError
    rmbn = rmbn_base
except ImportError:
    import robomimic.models.obs_core as rmbn  # newer robomimic

from mof.model.vision import crop_randomizer as dmvc


# Flag tensor our base-frame datasets inject; it is not a real observation
# and the encoder must not see it.
_FLAG_KEY = "_obs_in_base_frame"
_OBS_FILTER_ABS = (_FLAG_KEY,)
# In rel_action mode, base_pos / base_quat are kept in obs_dict for the
# world-frame reconstruction but filtered from the encoder input (matching
# the convention in `DiffusionUnetPolicyMoEDPRelTraj`).
_OBS_FILTER_REL_TRAJ = (_FLAG_KEY, "base_pos", "base_quat")


class TimmResNet18Conv(nn.Module):
    """Drop-in replacement for `robomimic.models.base_nets.ResNet18Conv`,
    backed by `timm.create_model('resnet18.tv_in1k', pretrained=True)`.

    Same I/O contract as ResNet18Conv:
      - input  (B, C, H, W)
      - output (B, 512, ceil(H/32), ceil(W/32))
      - exposes `output_shape(input_shape)` and an `nn.Sequential` named
        `nets` (the latter so robomimic introspection doesn't break).

    The Timm 'resnet18.tv_in1k' weights are the same torchvision-trained
    ImageNet weights `vision_models.resnet18(pretrained=True)` would load,
    so this matches the rest of our pipeline (`TimmObsEncoder`).
    """

    def __init__(self, input_channel: int = 3):
        super().__init__()
        net = timm.create_model("resnet18.tv_in1k", pretrained=True)
        if input_channel != 3:
            net.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self._input_channel = input_channel
        # Match robomimic: drop the last 2 children (avgpool, fc).
        self.nets = nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        assert len(input_shape) == 3
        return [
            512,
            int(math.ceil(input_shape[1] / 32.0)),
            int(math.ceil(input_shape[2] / 32.0)),
        ]

    def forward(self, x):
        return self.nets(x)


class _PreResizeRGBEncoder(nn.Module):
    """Thin wrapper that resizes RGB inputs to (size, size) before
    forwarding through a robomimic-built obs_encoder. Used to harmonize
    heterogeneous source resolutions (e.g. 224x224 bigym flipcup vs 84x84
    dexmimicgen / other bigym) so the downstream CropRandomizer +
    ResNet18 always operate on the resolution the encoder was built for.
    Without this, CropRandomizer(76) on a 224x224 input keeps only
    ~12% of the field-of-view.
    """

    def __init__(self, obs_encoder: nn.Module, size: int, rgb_keys):
        super().__init__()
        self.obs_encoder = obs_encoder
        self._rgb_keys = tuple(rgb_keys)
        self._size = size
        self._resize = T.Resize(size, antialias=True)

    def forward(self, obs_dict):
        new_obs = dict(obs_dict)
        for k in self._rgb_keys:
            if k in new_obs:
                new_obs[k] = self._resize(new_obs[k])
        return self.obs_encoder(new_obs)

    def output_shape(self):
        return self.obs_encoder.output_shape()


class DiffusionUnetMoEDPOfficial(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        pretrain: bool = True,
        train_action_key: str = "action",  # "action" (abs, official-faithful) or "rel_action"
        # When set, every RGB obs key is resized to (size, size) before the
        # robomimic encoder. obs_key_shapes is also overridden to (3, size,
        # size) so the encoder's CropRandomizer is built against the resized
        # resolution. Default None = no-op (paper-faithful). Use only for
        # tasks whose native image > 84x84 (e.g. flipcup at 224x224); a 76x76
        # CropRandomizer on raw 224x224 keeps only ~12% of the FOV.
        pre_resize_image_size: Optional[int] = None,
        # MoE_parameters
        use_MoE=False,
        n_emb: int = 256,
        p_drop_attn: float = 0.1,
        num_experts=6,
        top_k=2,
        boost: bool = False,
        use_aux_loss: bool = False,
        aux_loss_weight: float = 1.0,
        lambda_balance: float = 1.0,
        lambda_entropy: float = 0.1,
        gate_dim: int = 256,
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta — drop the obs-frame flag (and, in rel_action
        # mode, base_pos / base_quat too — they are kept in obs_dict for the
        # world-frame reconstruction but should not flow into the encoder).
        if train_action_key not in ("action", "rel_action"):
            raise ValueError(
                f"train_action_key must be 'action' or 'rel_action', got {train_action_key!r}"
            )
        self._train_action_key = train_action_key
        self._pred_action_norm_key = train_action_key
        self._encoder_drop_keys = (
            _OBS_FILTER_REL_TRAJ if train_action_key == "rel_action" else _OBS_FILTER_ABS
        )
        cprint(
            f"[Action] train_action_key={train_action_key!r}  encoder_drop_keys={self._encoder_drop_keys}",
            "cyan",
        )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = {
            k: v
            for k, v in shape_meta["obs"].items()
            if k not in self._encoder_drop_keys
        }
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                obs_config["rgb"].append(key)
            elif type_ == "low_dim":
                obs_config["low_dim"].append(key)

        # When pre_resize_image_size is set, override every RGB key's shape
        # so the robomimic encoder (CropRandomizer + ResNet18) is built
        # against the resized resolution. The actual image tensors are
        # resized at runtime by `_PreResizeRGBEncoder` wrapped around the
        # obs_encoder below.
        self._pre_resize_image_size = pre_resize_image_size
        if pre_resize_image_size is not None:
            for k in obs_config["rgb"]:
                c = obs_key_shapes[k][0]
                obs_key_shapes[k] = [c, pre_resize_image_size, pre_resize_image_size]
            cprint(
                f"[Encoder] pre_resize_image_size={pre_resize_image_size}: "
                f"RGB keys {tuple(obs_config['rgb'])} -> ({pre_resize_image_size}, "
                f"{pre_resize_image_size}); encoder built for this resolution.",
                "cyan",
            )

        # get raw robomimic config — verbatim from official.
        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        # === ONLY ENVIRONMENT-NECESSARY EDIT vs official (when pretrain=True) ===
        # When pretrain=True, replace robomimic's from-scratch ResNet18Conv
        # with a Timm-pretrained ResNet18 wrapper. We do this BEFORE the
        # BN->GN replacement below so the official sweep catches the Timm
        # backbone's BN layers too, producing a uniformly group-normalized
        # encoder. When pretrain=False, leave the encoder exactly as the
        # official release does (random-init ResNet18Conv).
        #
        # Note: robomimic's VisualCore keeps the backbone at TWO references
        # — `vc.backbone` (the named attribute) and `vc.nets[0]` (used by
        # forward via the Sequential). Both must be redirected to the same
        # new Timm module; the generic `replace_submodules` only updates
        # one reference and the assertion fails. Walk VisualCore parents
        # explicitly.
        self.pretrain = pretrain
        if pretrain:
            for _name, vc in obs_encoder.named_modules():
                if not (
                    hasattr(vc, "backbone")
                    and isinstance(vc.backbone, rmbn_base.ResNet18Conv)
                ):
                    continue
                old = vc.backbone
                new = TimmResNet18Conv(input_channel=old._input_channel)
                vc.backbone = new
                if hasattr(vc, "nets") and isinstance(vc.nets, nn.Sequential):
                    for i, child in enumerate(vc.nets):
                        if child is old:
                            vc.nets[i] = new
            cprint("[Encoder] pretrain=True: swapped robomimic ResNet18 -> Timm tv_in1k pretrained", "cyan")
        else:
            cprint("[Encoder] pretrain=False: using robomimic from-scratch ResNet18 (official-faithful)", "cyan")

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        # Wrap with pre-encoder transforms if requested. The wrappers
        # preserve the encoder's output_shape contract; obs_feature_dim
        # below is unchanged.
        if pre_resize_image_size is not None:
            obs_encoder = _PreResizeRGBEncoder(
                obs_encoder,
                size=pre_resize_image_size,
                rgb_keys=tuple(obs_config["rgb"]),
            )

        # create diffusion model — verbatim from official.
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.use_MoE = use_MoE
        if self.use_MoE is True:
            self.num_experts = num_experts
            self.top_k = top_k

            self.boost = boost
            self.use_aux_loss = use_aux_loss
            self.aux_loss_weight = aux_loss_weight

            self.cond_obs_emb = nn.Linear(obs_feature_dim, n_emb)
            obs_feature_dim = n_emb

            self.lambda_balance = lambda_balance
            self.lambda_entropy = lambda_entropy

            self.MoE_encoder = MoE(
                num_experts=num_experts,
                top_k=top_k,
                input_dim=n_emb,
                output_dim=n_emb,
                gate_dim=gate_dim,
                hidden_dim=4 * n_emb,
                dropout=p_drop_attn,
                boost=boost,
                use_aux_loss=use_aux_loss,
                lambda_balance=lambda_balance,
                lambda_entropy=lambda_entropy,
            )
            cprint(f"[MoE] use_MoE: {self.use_MoE}", "yellow")
            cprint(f"[MoE Config] lambda_balance: {self.lambda_balance}", "yellow")
            cprint(f"[MoE Config] lambda_entropy: {self.lambda_entropy}", "yellow")

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
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
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
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
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Rotation transforms — only used in `train_action_key='rel_action'`
        # mode for the world-frame action reconstruction. Cheap to construct
        # so we always instantiate.
        self.sixd_to_mat = RotationTransformer(
            from_rep="rotation_6d", to_rep="matrix"
        )
        self.quat_to_mat = RotationTransformer(
            from_rep="quaternion", to_rep="matrix"
        )
        self.mat_to_quat = RotationTransformer(
            from_rep="matrix", to_rep="quaternion"
        )

        # For workspace logging via get_last_train_metrics().
        self._last_train_metrics: Dict[str, float] = {}

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ============== inference  ==============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
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
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _drop_obs_keys(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v for k, v in obs_dict.items() if k not in self._encoder_drop_keys
        }

    # ---- frame conversion helpers (used only when train_action_key='rel_action').
    # Verbatim from `DiffusionUnetPolicyMoEDPRelTraj`. ----

    def _is_obs_in_base_frame(self, obs_dict: Dict[str, torch.Tensor]) -> bool:
        if _FLAG_KEY not in obs_dict:
            return False
        flag = obs_dict[_FLAG_KEY]
        if isinstance(flag, torch.Tensor):
            return bool((flag > 0).all().item())
        return bool(flag)

    def _get_base_frame_obs(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        obs_in_base_frame = self._is_obs_in_base_frame(obs_dict)
        out = dict(obs_dict)
        if _FLAG_KEY in out:
            out.pop(_FLAG_KEY)
        if obs_in_base_frame:
            return out
        if "base_pos" not in obs_dict or "base_quat" not in obs_dict:
            missing = [k for k in ("base_pos", "base_quat") if k not in obs_dict]
            raise KeyError(
                f"Missing required key(s) for base-frame conversion: {', '.join(missing)}"
            )
        base_pos = obs_dict["base_pos"]
        base_quat = obs_dict["base_quat"]
        base_rot = self.quat_to_mat.forward(base_quat)
        B, T = base_pos.shape[:2]
        eye = torch.eye(4, dtype=base_pos.dtype, device=base_pos.device)
        base_T = eye.view(1, 1, 4, 4).repeat(B, T, 1, 1)
        base_T[:, :, :3, :3] = base_rot
        base_T[:, :, :3, 3] = base_pos
        base_T_inv = torch.linalg.inv(base_T)
        for key in ("left_ee_pos", "right_ee_pos", "head_site_pos"):
            if key not in out:
                continue
            pos = out[key]
            pos_T = eye.view(1, 1, 4, 4).repeat(B, T, 1, 1)
            pos_T[:, :, :3, 3] = pos
            rel_pos_T = torch.matmul(base_T_inv, pos_T)
            out[key] = rel_pos_T[:, :, :3, 3]
        for key in ("left_ee_quat", "right_ee_quat", "head_site_quat"):
            if key not in out:
                continue
            quat = out[key]
            rot = self.quat_to_mat.forward(quat)
            quat_T = eye.view(1, 1, 4, 4).repeat(B, T, 1, 1)
            quat_T[:, :, :3, :3] = rot
            rel_quat_T = torch.matmul(base_T_inv, quat_T)
            out[key] = self.mat_to_quat.forward(rel_quat_T[:, :, :3, :3])
        out["base_pos"] = base_pos
        out["base_quat"] = base_quat
        return out

    def _reconstruct_abs_action_from_rel_trans(
        self,
        rel_action_pred: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, T = rel_action_pred.shape[:2]
        device = rel_action_pred.device
        left_xyz = rel_action_pred[:, :, :3]
        left_rot_6d = rel_action_pred[:, :, 3:9]
        left_rot = self.sixd_to_mat.forward(left_rot_6d)
        left_rel_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        left_rel_T[:, :, :3, :3] = left_rot
        left_rel_T[:, :, :3, 3] = left_xyz
        left_cur_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        left_cur_rot = self.quat_to_mat.forward(obs_dict["left_ee_quat"][:, -1:])
        left_cur_trans = obs_dict["left_ee_pos"][:, -1:]
        left_cur_T[:, :, :3, :3] = left_cur_rot
        left_cur_T[:, :, :3, 3] = left_cur_trans
        left_abs_base_T = torch.matmul(left_cur_T, left_rel_T)
        right_xyz = rel_action_pred[:, :, 9:12]
        right_rot_6d = rel_action_pred[:, :, 12:18]
        right_rot = self.sixd_to_mat.forward(right_rot_6d)
        right_rel_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        right_rel_T[:, :, :3, :3] = right_rot
        right_rel_T[:, :, :3, 3] = right_xyz
        right_cur_T = torch.eye(4).repeat(B, T, 1, 1).to(device)
        right_cur_rot = self.quat_to_mat.forward(obs_dict["right_ee_quat"][:, -1:])
        right_cur_trans = obs_dict["right_ee_pos"][:, -1:]
        right_cur_T[:, :, :3, :3] = right_cur_rot
        right_cur_T[:, :, :3, 3] = right_cur_trans
        right_abs_base_T = torch.matmul(right_cur_T, right_rel_T)
        if "base_pos" not in obs_dict or "base_quat" not in obs_dict:
            missing = [k for k in ("base_pos", "base_quat") if k not in obs_dict]
            raise KeyError(
                f"Missing required key(s) for abs-action reconstruction: {', '.join(missing)}"
            )
        base_pos_world = obs_dict["base_pos"][:, -1:]
        base_quat_world = obs_dict["base_quat"][:, -1:]
        base_rot_world = self.quat_to_mat.forward(base_quat_world)
        base_T = torch.eye(4, dtype=rel_action_pred.dtype, device=device).view(1, 1, 4, 4).repeat(B, T, 1, 1)
        base_T[:, :, :3, :3] = base_rot_world.expand(-1, T, -1, -1)
        base_T[:, :, :3, 3] = base_pos_world.expand(-1, T, -1)
        left_abs_T = torch.matmul(base_T, left_abs_base_T)
        right_abs_T = torch.matmul(base_T, right_abs_base_T)
        left_abs_rot_6d = self.sixd_to_mat.inverse(left_abs_T[:, :, :3, :3])
        right_abs_rot_6d = self.sixd_to_mat.inverse(right_abs_T[:, :, :3, :3])
        abs_action = torch.cat(
            [
                left_abs_T[:, :, :3, 3],
                left_abs_rot_6d,
                right_abs_T[:, :, :3, 3],
                right_abs_rot_6d,
                rel_action_pred[:, :, 18:],
            ],
            dim=-1,
        )
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

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], metrics=None
    ) -> Dict[str, torch.Tensor]:
        """Verbatim from official, with two extensions for our env:
          - `_drop_obs_keys` filters the obs frame flag (and, in rel_action
            mode, base_pos / base_quat) before the encoder sees them.
          - When `train_action_key='rel_action'`, obs are first transformed
            into base frame and the unnormalized prediction is reconstructed
            into a world-frame absolute action before returning to the env.
        """
        assert "past_action" not in obs_dict
        if self._train_action_key == "rel_action":
            obs_model = self._get_base_frame_obs(obs_dict)
        else:
            obs_model = obs_dict
        nobs = self.normalizer.normalize(self._drop_obs_keys(obs_model))
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            if self.use_MoE is True:
                nobs_features_embedding = self.cond_obs_emb(nobs_features).reshape(
                    B * self.n_obs_steps, -1
                )
                nobs_MoE_features, aux_loss, aux_loss_dict = self.MoE_encoder(
                    x=nobs_features_embedding, metrics=metrics
                )
                nobs_features = nobs_MoE_features.reshape(B, self.n_obs_steps, -1)

            global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer[self._pred_action_norm_key].unnormalize(naction_pred)
        if self._train_action_key == "rel_action":
            # rel_action -> world-frame absolute, using the (un-filtered)
            # base-frame obs (which retains base_pos / base_quat for the
            # final base-frame -> world-frame multiply).
            action_pred = self._reconstruct_abs_action_from_rel_trans(
                action_pred, obs_model
            )

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def get_MoE_encode_output(
        self, obs_dict: Dict[str, torch.Tensor], metrics=None
    ) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(self._drop_obs_keys(obs_dict))
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features_embedding = self.cond_obs_emb(nobs_features).reshape(
            B * self.n_obs_steps, -1
        )
        nobs_MoE_features, _, _ = self.MoE_encoder(
            x=nobs_features_embedding, metrics=metrics
        )
        return nobs_MoE_features

    # ============== training  ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_last_train_metrics(self):
        return dict(self._last_train_metrics)

    def compute_loss(self, batch):
        """Verbatim from official except:
          - filters dex-injected obs keys
          - returns just the scalar loss (stashes loss_dict for the workspace
            to retrieve via get_last_train_metrics()).
        """
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(self._drop_obs_keys(batch["obs"]))
        nactions = self.normalizer[self._train_action_key].normalize(
            batch[self._train_action_key]
        )
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = self.obs_encoder(this_nobs)
            if self.use_MoE is True:
                nobs_features_embedding = self.cond_obs_emb(nobs_features).reshape(
                    batch_size * self.n_obs_steps, -1
                )
                nobs_MoE_features, aux_loss, aux_loss_dict = self.MoE_encoder(
                    x=nobs_features_embedding
                )
                nobs_features = nobs_MoE_features.reshape(
                    batch_size, self.n_obs_steps, -1
                )
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
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
        loss = loss.mean()

        if self.use_aux_loss:
            bc_loss = loss.detach().clone()
            loss = loss + self.aux_loss_weight * aux_loss
            # Stash detached tensors so the workspace's _collect_train_metrics
            # picks them up — its filter requires `isinstance(value, torch.Tensor)`
            # and silently drops non-tensor values. Earlier versions stored
            # Python floats via `.item()` which broke wandb logging.
            loss_dict = {
                "bc_loss": bc_loss.detach(),
                "total_aux_loss": aux_loss.detach(),
                "total_loss": loss.detach(),
            }
            if aux_loss_dict is not None:
                for k, v in aux_loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_dict[k] = v.detach()
                    else:
                        # scalar coefficients (lambda_balance, lambda_entropy)
                        loss_dict[k] = torch.tensor(float(v))
        else:
            loss_dict = {"bc_loss": loss.detach()}

        # Stash for workspace logger; return scalar to match our workspace API.
        self._last_train_metrics = {
            f"moe_official/{k}": v for k, v in loss_dict.items()
        }
        return loss
