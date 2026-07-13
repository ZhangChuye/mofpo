from mof.model.common.normalizer import SingleFieldLinearNormalizer
from mof.common.pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split
import numpy as np

def _compute_range_scale_offset(input_min, input_max, output_max=1, output_min=-1, range_eps=1e-7):
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range = input_range.copy()
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return scale, offset


def _compute_xy_symmetric_scale_offset(
    input_min,
    input_max,
    output_max=1,
    output_min=-1,
    range_eps=1e-7,
    shared_xy_half_range=None,
    shared_xy_center=None,
):
    scale, offset = _compute_range_scale_offset(
        input_min=input_min,
        input_max=input_max,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
    )
    symmetric_dims = min(2, input_min.shape[-1])
    if symmetric_dims == 0:
        return scale, offset

    xy_min = input_min[:symmetric_dims]
    xy_max = input_max[:symmetric_dims]
    if shared_xy_center is None:
        xy_center = 0.5 * (xy_max + xy_min)
    else:
        xy_center = np.asarray(shared_xy_center[:symmetric_dims], dtype=input_min.dtype)
    xy_half_range = 0.5 * (xy_max - xy_min)
    shared_half_range = (
        np.max(xy_half_range) if shared_xy_half_range is None else shared_xy_half_range
    )

    output_mid = (output_max + output_min) / 2
    output_half = (output_max - output_min) / 2
    if shared_half_range < range_eps:
        xy_scale = np.ones_like(xy_center)
    else:
        xy_scale = np.full_like(xy_center, output_half / shared_half_range)
    xy_offset = output_mid - xy_scale * xy_center

    scale[:symmetric_dims] = xy_scale
    offset[:symmetric_dims] = xy_offset
    return scale, offset

def _compute_xyz_symmetric_scale_offset(
    input_min,
    input_max,
    output_max=1,
    output_min=-1,
    range_eps=1e-7,
    shared_xyz_half_range=None,
    shared_xyz_center=None,
):
    scale, offset = _compute_range_scale_offset(
        input_min=input_min,
        input_max=input_max,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
    )
    symmetric_dims = min(3, input_min.shape[-1])
    if symmetric_dims == 0:
        return scale, offset

    xyz_min = input_min[:symmetric_dims]
    xyz_max = input_max[:symmetric_dims]
    if shared_xyz_center is None:
        xyz_center = 0.5 * (xyz_max + xyz_min)
    else:
        xyz_center = np.asarray(shared_xyz_center[:symmetric_dims], dtype=input_min.dtype)
    xyz_half_range = 0.5 * (xyz_max - xyz_min)
    shared_half_range = (
        np.max(xyz_half_range) if shared_xyz_half_range is None else shared_xyz_half_range
    )

    output_mid = (output_max + output_min) / 2
    output_half = (output_max - output_min) / 2
    if shared_half_range < range_eps:
        xyz_scale = np.ones_like(xyz_center)
    else:
        xyz_scale = np.full_like(xyz_center, output_half / shared_half_range)
    xyz_offset = output_mid - xyz_scale * xyz_center

    scale[:symmetric_dims] = xyz_scale
    offset[:symmetric_dims] = xyz_offset
    return scale, offset

def _compute_shared_xy_half_range(stats, range_eps=1e-7):
    half_ranges = []
    for stat in stats:
        dims = min(2, stat["min"].shape[-1])
        if dims == 0:
            continue
        half = 0.5 * (stat["max"][:dims] - stat["min"][:dims])
        half_ranges.append(half)
    if not half_ranges:
        return range_eps
    return max(float(np.max(half)) for half in half_ranges)


def _compute_shared_xy_center_and_half_range(stats, range_eps=1e-7):
    mins = []
    maxs = []
    for stat in stats:
        dims = min(2, stat["min"].shape[-1])
        if dims == 0:
            continue
        mins.append(stat["min"][:dims])
        maxs.append(stat["max"][:dims])
    if not mins:
        return None, range_eps

    mins = np.stack(mins, axis=0)
    maxs = np.stack(maxs, axis=0)
    shared_xy_min = np.min(mins, axis=0)
    shared_xy_max = np.max(maxs, axis=0)
    shared_xy_center = 0.5 * (shared_xy_min + shared_xy_max)
    shared_xy_half_range = np.max(0.5 * (shared_xy_max - shared_xy_min))
    return shared_xy_center, max(shared_xy_half_range, range_eps)


def _compute_shared_xyz_half_range(stats, range_eps=1e-7):
    half_ranges = []
    for stat in stats:
        dims = min(3, stat["min"].shape[-1])
        if dims == 0:
            continue
        half = 0.5 * (stat["max"][:dims] - stat["min"][:dims])
        half_ranges.append(half)
    if not half_ranges:
        return range_eps
    return max(float(np.max(half)) for half in half_ranges)


def _compute_shared_xyz_center_and_half_range(stats, range_eps=1e-7):
    mins = []
    maxs = []
    for stat in stats:
        dims = min(3, stat["min"].shape[-1])
        if dims == 0:
            continue
        mins.append(stat["min"][:dims])
        maxs.append(stat["max"][:dims])
    if not mins:
        return None, range_eps

    mins = np.stack(mins, axis=0)
    maxs = np.stack(maxs, axis=0)
    shared_xyz_min = np.min(mins, axis=0)
    shared_xyz_max = np.max(maxs, axis=0)
    shared_xyz_center = 0.5 * (shared_xyz_min + shared_xyz_max)
    shared_xyz_half_range = np.max(0.5 * (shared_xyz_max - shared_xyz_min))
    return shared_xyz_center, max(shared_xyz_half_range, range_eps)


def get_image_identity_normalizer():
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max'].copy()
    input_min = stat['min'].copy()
    scale, offset = _compute_range_scale_offset(
        input_min=input_min,
        input_max=input_max,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_range_symmetric_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization with shared xy scale and per-dim center offset
    input_max = stat['max'].copy()
    input_min = stat['min'].copy()
    scale, offset = _compute_xy_symmetric_scale_offset(
        input_min=input_min,
        input_max=input_max,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def get_range_symmetric_xyz_normalizer_from_stat(
    stat, output_max=1, output_min=-1, range_eps=1e-7
):
    input_max = stat['max'].copy()
    input_min = stat['min'].copy()
    scale, offset = _compute_xyz_symmetric_scale_offset(
        input_min=input_min,
        input_max=input_max,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_voxel_identity_normalizer():
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_point_cloud_identity_normalizer(dim=6):
    scale = np.array(dim*[1], dtype=np.float32)
    offset = np.array(dim*[0], dtype=np.float32)
    stat = {
        'min': np.array(dim*[-1], dtype=np.float32),
        'max': np.array(dim*[1], dtype=np.float32),
        'mean': np.array(dim*[0], dtype=np.float32),
        'std': np.array(dim*[np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_range_normalizer():
    """Maps [0, 1] -> [-1, 1]. Use when encoder does not apply its own normalization."""
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def get_image_identity_normalizer():
    """No-op for RGB (scale=1, offset=0). Use when encoder applies ImageNet etc. so data should stay in [0, 1]."""
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'rot': x[...,3:6],
            'gripper': x[...,6:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat['mean'])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info
    
    def get_gripper_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    rot_param, rot_info = get_rot_param_info(result['rot'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def bigym_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos1': x[...,:3],
            'rot1': x[...,3:9],
            'pos2': x[...,9:12],
            'rot2': x[...,12:18],
            'gripper': x[...,18:],
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat
    
    def get_rot_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    def get_gripper_param_info(stat, output_max=1, output_min=-1):
        # -1, 1 normalization
        stat['max'][:] = 1
        stat['min'][:] = 0
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min

        return {'scale': scale, 'offset': offset}, stat

    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    rot1_param, rot1_info = get_rot_param_info(result['rot1'])
    pos2_param, pos2_info = get_pos_param_info(result['pos2'])
    rot2_param, rot2_info = get_rot_param_info(result['rot2'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos1_param, rot1_param, pos2_param, rot2_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos1_info, rot1_info, pos2_info, rot2_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robosuite_dual_arm_action_normalizer_from_stat(stat):
    """Normalizer for robosuite dual-arm 20D actions with [-1, 1] grippers.

    Same as bigym_action_only_normalizer_from_stat but assumes gripper
    input range [-1, 1] (robosuite) instead of [0, 1] (BigYM).
    """
    result = dict_apply_split(
        stat, lambda x: {
            'pos1': x[...,:3],
            'rot1': x[...,3:9],
            'pos2': x[...,9:12],
            'rot2': x[...,12:18],
            'gripper': x[...,18:],
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    def get_gripper_param_info(stat, output_max=1, output_min=-1):
        # Robosuite grippers are in [-1, 1] → identity normalization
        stat['max'][:] = 1
        stat['min'][:] = -1
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        return {'scale': scale, 'offset': offset}, stat

    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    rot1_param, rot1_info = get_rot_param_info(result['rot1'])
    pos2_param, pos2_info = get_pos_param_info(result['pos2'])
    rot2_param, rot2_info = get_rot_param_info(result['rot2'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos1_param, rot1_param, pos2_param, rot2_param, gripper_param],
        lambda x: np.concatenate(x, axis=-1))
    info = dict_apply_reduce(
        [pos1_info, rot1_info, pos2_info, rot2_info, gripper_info],
        lambda x: np.concatenate(x, axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robosuite_dual_arm_action_normalizer_dex_from_stat(stat):
    """Normalizer for robosuite dual-arm dex-hand actions (action_dim = 18 + 2*grip_dim).

    Layout per arm: [pos(3), rot6d(6)]; then [left_grip(grip_dim), right_grip(grip_dim)].
    Differs from `robosuite_dual_arm_action_normalizer_from_stat` (parallel-jaw) in
    that the gripper channels are joint-space targets (e.g. dex hand qpos in [0, π/2])
    that fall outside [-1, 1]. We range-normalize them from actual stats so the
    diffusion policy's `clip_sample=True` does not silently clamp valid commands.
    """
    result = dict_apply_split(
        stat, lambda x: {
            'pos1': x[...,:3],
            'rot1': x[...,3:9],
            'pos2': x[...,9:12],
            'rot2': x[...,12:18],
            'gripper': x[...,18:],
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    def get_gripper_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # Range-normalize per-channel from actual data (dex hand qpos targets).
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        return {'scale': scale, 'offset': offset}, stat

    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    rot1_param, rot1_info = get_rot_param_info(result['rot1'])
    pos2_param, pos2_info = get_pos_param_info(result['pos2'])
    rot2_param, rot2_info = get_rot_param_info(result['rot2'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos1_param, rot1_param, pos2_param, rot2_param, gripper_param],
        lambda x: np.concatenate(x, axis=-1))
    info = dict_apply_reduce(
        [pos1_info, rot1_info, pos2_info, rot2_info, gripper_info],
        lambda x: np.concatenate(x, axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def bigym_action_only_symmetric_normalizer_from_stat(
    stat, share_center_for_arms=False
):
    result = dict_apply_split(
        stat, lambda x: {
            'pos1': x[...,:3],
            'rot1': x[...,3:9],
            'pos2': x[...,9:12],
            'rot2': x[...,12:18],
            'gripper': x[...,18:],
    })

    if share_center_for_arms:
        shared_xy_center, shared_xy_half_range = _compute_shared_xy_center_and_half_range(
            [result["pos1"], result["pos2"]]
        )
    else:
        shared_xy_center = None
        shared_xy_half_range = _compute_shared_xy_half_range(
            [result["pos1"], result["pos2"]]
        )

    def get_pos_param_info(
        stat,
        output_max=1,
        output_min=-1,
        range_eps=1e-7,
        shared_xy_half_range=shared_xy_half_range,
        shared_xy_center=shared_xy_center,
    ):
        # symmetric xy normalization in output space with center-preserving offset
        input_max = stat['max'].copy()
        input_min = stat['min'].copy()
        scale, offset = _compute_xy_symmetric_scale_offset(
            input_min=input_min,
            input_max=input_max,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            shared_xy_half_range=shared_xy_half_range,
            shared_xy_center=shared_xy_center,
        )

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    def get_gripper_param_info(stat, output_max=1, output_min=-1):
        # -1, 1 normalization
        stat['max'][:] = 1
        stat['min'][:] = 0
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min

        return {'scale': scale, 'offset': offset}, stat

    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    rot1_param, rot1_info = get_rot_param_info(result['rot1'])
    pos2_param, pos2_info = get_pos_param_info(result['pos2'])
    rot2_param, rot2_info = get_rot_param_info(result['rot2'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos1_param, rot1_param, pos2_param, rot2_param, gripper_param],
        lambda x: np.concatenate(x, axis=-1))
    info = dict_apply_reduce(
        [pos1_info, rot1_info, pos2_info, rot2_info, gripper_info],
        lambda x: np.concatenate(x, axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def bigym_action_only_symmetric_xyz_normalizer_from_stat(
    stat, share_center_for_arms=False
):
    result = dict_apply_split(
        stat, lambda x: {
            'pos1': x[...,:3],
            'rot1': x[...,3:9],
            'pos2': x[...,9:12],
            'rot2': x[...,12:18],
            'gripper': x[...,18:],
    })

    if share_center_for_arms:
        shared_xyz_center, shared_xyz_half_range = _compute_shared_xyz_center_and_half_range(
            [result["pos1"], result["pos2"]]
        )
    else:
        shared_xyz_center = None
        shared_xyz_half_range = _compute_shared_xyz_half_range(
            [result["pos1"], result["pos2"]]
        )

    def get_pos_param_info(
        stat,
        output_max=1,
        output_min=-1,
        range_eps=1e-7,
        shared_xyz_half_range=shared_xyz_half_range,
        shared_xyz_center=shared_xyz_center,
    ):
        input_max = stat['max'].copy()
        input_min = stat['min'].copy()
        scale, offset = _compute_xyz_symmetric_scale_offset(
            input_min=input_min,
            input_max=input_max,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            shared_xyz_half_range=shared_xyz_half_range,
            shared_xyz_center=shared_xyz_center,
        )

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    def get_gripper_param_info(stat, output_max=1, output_min=-1):
        stat['max'][:] = 1
        stat['min'][:] = 0
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min

        return {'scale': scale, 'offset': offset}, stat

    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    rot1_param, rot1_info = get_rot_param_info(result['rot1'])
    pos2_param, pos2_info = get_pos_param_info(result['pos2'])
    rot2_param, rot2_info = get_rot_param_info(result['rot2'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos1_param, rot1_param, pos2_param, rot2_param, gripper_param],
        lambda x: np.concatenate(x, axis=-1))
    info = dict_apply_reduce(
        [pos1_info, rot1_info, pos2_info, rot2_info, gripper_info],
        lambda x: np.concatenate(x, axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robosuite_dual_arm_action_symmetric_xyz_normalizer_from_stat(
    stat, share_center_for_arms=False
):
    """Like bigym_action_only_symmetric_xyz_normalizer_from_stat but with [-1, 1] grippers."""
    result = dict_apply_split(
        stat, lambda x: {
            'pos1': x[...,:3],
            'rot1': x[...,3:9],
            'pos2': x[...,9:12],
            'rot2': x[...,12:18],
            'gripper': x[...,18:],
    })

    if share_center_for_arms:
        shared_xyz_center, shared_xyz_half_range = _compute_shared_xyz_center_and_half_range(
            [result["pos1"], result["pos2"]]
        )
    else:
        shared_xyz_center = None
        shared_xyz_half_range = _compute_shared_xyz_half_range(
            [result["pos1"], result["pos2"]]
        )

    def get_pos_param_info(
        stat,
        output_max=1,
        output_min=-1,
        range_eps=1e-7,
        shared_xyz_half_range=shared_xyz_half_range,
        shared_xyz_center=shared_xyz_center,
    ):
        input_max = stat['max'].copy()
        input_min = stat['min'].copy()
        scale, offset = _compute_xyz_symmetric_scale_offset(
            input_min=input_min,
            input_max=input_max,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            shared_xyz_half_range=shared_xyz_half_range,
            shared_xyz_center=shared_xyz_center,
        )
        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    def get_gripper_param_info(stat, output_max=1, output_min=-1):
        stat['max'][:] = 1
        stat['min'][:] = -1
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        return {'scale': scale, 'offset': offset}, stat

    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    rot1_param, rot1_info = get_rot_param_info(result['rot1'])
    pos2_param, pos2_info = get_pos_param_info(result['pos2'])
    rot2_param, rot2_info = get_rot_param_info(result['rot2'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos1_param, rot1_param, pos2_param, rot2_param, gripper_param],
        lambda x: np.concatenate(x, axis=-1))
    info = dict_apply_reduce(
        [pos1_info, rot1_info, pos2_info, rot2_info, gripper_info],
        lambda x: np.concatenate(x, axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_symmetric_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        abs_max = np.max([np.abs(stat['max'][:3]), np.abs(stat['min'][:3])])
        input_max[:3] = abs_max
        input_min[:3] = -abs_max
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat['max'].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat, lambda x: {
            'pos0': x[...,:3],
            'other0': x[...,3:Dah],
            'pos1': x[...,Dah:Dah+3],
            'other1': x[...,Dah+3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos0_param, pos0_info = get_pos_param_info(result['pos0'])
    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    other0_param, other0_info = get_other_param_info(result['other0'])
    other1_param, other1_info = get_other_param_info(result['other1'])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat
