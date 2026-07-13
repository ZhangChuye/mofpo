from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from mof.model.common.rotation_transformer import RotationTransformer


LEFT_POS_SLICE = slice(0, 3)
LEFT_ROT_SLICE = slice(3, 9)
RIGHT_POS_SLICE = slice(9, 12)
RIGHT_ROT_SLICE = slice(12, 18)
# Defaults for parallel-jaw (1-D gripper per arm, total action_dim=20).
# For dex-hand (6-D gripper per arm, action_dim=30) use gripper_slices() / arm_channels().
LEFT_GRIPPER_SLICE = slice(18, 19)
RIGHT_GRIPPER_SLICE = slice(19, 20)

LEFT_ARM_CHANNELS = tuple(range(0, 9)) + (18,)
RIGHT_ARM_CHANNELS = tuple(range(9, 18)) + (19,)
POSE_POSITION_KEYS = ("left_ee_pos", "right_ee_pos", "head_site_pos")
POSE_QUAT_KEYS = ("left_ee_quat", "right_ee_quat", "head_site_quat")


def gripper_slices(action_dim: int):
    """Return (left_slice, right_slice) into the gripper channels of an action.

    Layout: action_dim = 18 + 2 * gripper_dim, where gripper_dim is per-arm.
    """
    grip_dim = (action_dim - 18) // 2
    return slice(18, 18 + grip_dim), slice(18 + grip_dim, 18 + 2 * grip_dim)


def arm_channels(action_dim: int):
    """Return (left_channels, right_channels) covering pose + gripper for each arm."""
    grip_dim = (action_dim - 18) // 2
    left = tuple(range(0, 9)) + tuple(range(18, 18 + grip_dim))
    right = tuple(range(9, 18)) + tuple(range(18 + grip_dim, 18 + 2 * grip_dim))
    return left, right


def rotation_6d_columns_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 matrix (column convention).

    The 6 values are interpreted as two 3-vectors that become the first two
    *columns* of the rotation matrix after Gram-Schmidt orthogonalisation.
    This convention is required so that ``rotate_raw_vectors_between_frames``
    can treat each 3-vector block independently (R @ vec).
    """
    col1 = F.normalize(d6[..., 0:3], dim=-1)
    col2_raw = d6[..., 3:6]
    col2 = col2_raw - (col1 * col2_raw).sum(dim=-1, keepdim=True) * col1
    col2 = F.normalize(col2, dim=-1)
    col3 = torch.cross(col1, col2, dim=-1)
    return torch.stack((col1, col2, col3), dim=-1)


def matrix_to_rotation_6d_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation (column convention)."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


# ---------------------------------------------------------------------------
# Boundary conversion: pytorch3d row convention <-> internal column convention
# ---------------------------------------------------------------------------

def action_rot6d_row_to_column(action: torch.Tensor) -> torch.Tensor:
    """Convert rot6d channels in a 20D action from pytorch3d row to column convention.

    Call this when ingesting world-frame actions from the replay buffer (which
    stores rot6d in pytorch3d row convention) into the MoE pipeline (which uses
    column convention internally).
    """
    out = action.clone()
    for rot_slice in (LEFT_ROT_SLICE, RIGHT_ROT_SLICE):
        d6 = action[..., rot_slice]
        # Parse with row convention (Gram-Schmidt on rows)
        a1 = F.normalize(d6[..., 0:3], dim=-1)
        a2_raw = d6[..., 3:6]
        a2 = a2_raw - (a1 * a2_raw).sum(dim=-1, keepdim=True) * a1
        a2 = F.normalize(a2, dim=-1)
        a3 = torch.cross(a1, a2, dim=-1)
        R = torch.stack((a1, a2, a3), dim=-2)  # rows → (..., 3, 3)
        # Re-encode as column convention
        out[..., rot_slice] = matrix_to_rotation_6d_columns(R)
    return out


def action_rot6d_column_to_row(action: torch.Tensor) -> torch.Tensor:
    """Convert rot6d channels in a 20D action from column to pytorch3d row convention.

    Call this when outputting world-frame predictions from the MoE pipeline
    back to the external world (which expects pytorch3d row convention).
    """
    out = action.clone()
    for rot_slice in (LEFT_ROT_SLICE, RIGHT_ROT_SLICE):
        R = rotation_6d_columns_to_matrix(action[..., rot_slice])
        # Encode as row convention: first two rows of R
        out[..., rot_slice] = torch.cat((R[..., 0, :], R[..., 1, :]), dim=-1)
    return out


def eye_transform(*shape: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ones = (1,) * len(shape)
    return torch.eye(4, dtype=dtype, device=device).view(*ones, 4, 4).repeat(*shape, 1, 1)


def pose_to_transform(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    T = eye_transform(*pos.shape[:-1], dtype=pos.dtype, device=pos.device)
    T[..., :3, :3] = rot
    T[..., :3, 3] = pos
    return T


def world_frame_to_transform(
    pos: torch.Tensor,
    quat: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    return pose_to_transform(pos, quat_to_mat.forward(quat))


def _expand_time_dim(T: torch.Tensor, horizon: int) -> torch.Tensor:
    if T.ndim == 3:
        T = T.unsqueeze(1)
    if T.shape[1] == horizon:
        return T
    if T.shape[1] != 1:
        raise ValueError(f"Cannot expand transform with shape {T.shape} to horizon {horizon}.")
    return T.expand(-1, horizon, -1, -1)


def transform_points(frame_T_other: torch.Tensor, pts_other: torch.Tensor) -> torch.Tensor:
    rot = frame_T_other[..., :3, :3]
    trans = frame_T_other[..., :3, 3]
    return torch.einsum("...ij,...j->...i", rot, pts_other) + trans


def split_action_pose_components(action: torch.Tensor) -> Dict[str, torch.Tensor]:
    left_grip, right_grip = gripper_slices(action.shape[-1])
    return {
        "left_pos": action[..., LEFT_POS_SLICE],
        "left_rot": rotation_6d_columns_to_matrix(action[..., LEFT_ROT_SLICE]),
        "right_pos": action[..., RIGHT_POS_SLICE],
        "right_rot": rotation_6d_columns_to_matrix(action[..., RIGHT_ROT_SLICE]),
        "left_gripper": action[..., left_grip],
        "right_gripper": action[..., right_grip],
    }


def assemble_action_pose_components(
    left_pos: torch.Tensor,
    left_rot: torch.Tensor,
    right_pos: torch.Tensor,
    right_rot: torch.Tensor,
    left_gripper: torch.Tensor,
    right_gripper: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            left_pos,
            matrix_to_rotation_6d_columns(left_rot),
            right_pos,
            matrix_to_rotation_6d_columns(right_rot),
            left_gripper,
            right_gripper,
        ],
        dim=-1,
    )


def convert_pose_action_between_frames(
    action: torch.Tensor,
    world_T_source_frame: torch.Tensor,
    world_T_target_frame: torch.Tensor,
) -> torch.Tensor:
    horizon = action.shape[1]
    world_T_source_frame = _expand_time_dim(world_T_source_frame, horizon)
    world_T_target_frame = _expand_time_dim(world_T_target_frame, horizon)
    target_T_source = torch.matmul(torch.linalg.inv(world_T_target_frame), world_T_source_frame)

    comps = split_action_pose_components(action)
    left_pos = transform_points(target_T_source, comps["left_pos"])
    right_pos = transform_points(target_T_source, comps["right_pos"])
    left_rot = torch.matmul(target_T_source[..., :3, :3], comps["left_rot"])
    right_rot = torch.matmul(target_T_source[..., :3, :3], comps["right_rot"])
    return assemble_action_pose_components(
        left_pos=left_pos,
        left_rot=left_rot,
        right_pos=right_pos,
        right_rot=right_rot,
        left_gripper=comps["left_gripper"],
        right_gripper=comps["right_gripper"],
    )


def convert_pose_action_between_frames_as_vectors(
    action: torch.Tensor,
    world_T_source_frame: torch.Tensor,
    world_T_target_frame: torch.Tensor,
) -> torch.Tensor:
    """Convert pose-like actions while treating rot6d columns as raw vectors.

    This is exact for noisy DDIM samples because the first 18 channels are
    transformed linearly: position gets the usual rigid transform and the two
    rot6d columns per arm are rotated as independent 3-vectors without
    projecting through Gram-Schmidt.
    """
    horizon = action.shape[1]
    world_T_source_frame = _expand_time_dim(world_T_source_frame, horizon)
    world_T_target_frame = _expand_time_dim(world_T_target_frame, horizon)
    target_T_source = torch.matmul(torch.linalg.inv(world_T_target_frame), world_T_source_frame)
    target_R_source = target_T_source[..., :3, :3]

    out = action.clone()
    out[..., LEFT_POS_SLICE] = transform_points(target_T_source, action[..., LEFT_POS_SLICE])
    out[..., RIGHT_POS_SLICE] = transform_points(target_T_source, action[..., RIGHT_POS_SLICE])

    left_rot_vecs = action[..., LEFT_ROT_SLICE].unflatten(-1, (2, 3))
    right_rot_vecs = action[..., RIGHT_ROT_SLICE].unflatten(-1, (2, 3))
    out[..., LEFT_ROT_SLICE] = torch.einsum(
        "...ij,...kj->...ki", target_R_source, left_rot_vecs
    ).flatten(-2, -1)
    out[..., RIGHT_ROT_SLICE] = torch.einsum(
        "...ij,...kj->...ki", target_R_source, right_rot_vecs
    ).flatten(-2, -1)
    return out


def rotate_raw_vectors_between_frames(
    delta: torch.Tensor,
    world_T_source_frame: torch.Tensor,
    world_T_target_frame: torch.Tensor,
) -> torch.Tensor:
    """Rotate all six 3-vector blocks (pos + raw rot6d columns) without Gram-Schmidt.

    Use this for noise/epsilon quantities whose rotation channels are arbitrary
    values, not valid rotation columns.  Unlike ``convert_pose_delta_between_frames``
    this avoids the lossy Gram-Schmidt projection and gives an exact roundtrip.
    """
    horizon = delta.shape[1]
    world_T_source_frame = _expand_time_dim(world_T_source_frame, horizon)
    world_T_target_frame = _expand_time_dim(world_T_target_frame, horizon)
    target_R_source = torch.matmul(
        torch.linalg.inv(world_T_target_frame[..., :3, :3]),
        world_T_source_frame[..., :3, :3],
    )
    # Treat the first 18 dims as 6 independent 3-vectors and rotate each one
    vecs = delta[..., :18].unflatten(-1, (6, 3))  # (..., 6, 3)
    rotated = torch.einsum("...ij,...kj->...ki", target_R_source, vecs)  # (..., 6, 3)
    return torch.cat([rotated.flatten(-2, -1), delta[..., 18:]], dim=-1)


def convert_pose_delta_between_frames(
    delta: torch.Tensor,
    world_T_source_frame: torch.Tensor,
    world_T_target_frame: torch.Tensor,
) -> torch.Tensor:
    horizon = delta.shape[1]
    world_T_source_frame = _expand_time_dim(world_T_source_frame, horizon)
    world_T_target_frame = _expand_time_dim(world_T_target_frame, horizon)
    target_R_source = torch.matmul(
        torch.linalg.inv(world_T_target_frame[..., :3, :3]),
        world_T_source_frame[..., :3, :3],
    )
    comps = split_action_pose_components(delta)
    left_pos = torch.einsum("...ij,...j->...i", target_R_source, comps["left_pos"])
    right_pos = torch.einsum("...ij,...j->...i", target_R_source, comps["right_pos"])
    left_rot = torch.matmul(target_R_source, comps["left_rot"])
    right_rot = torch.matmul(target_R_source, comps["right_rot"])
    return assemble_action_pose_components(
        left_pos=left_pos,
        left_rot=left_rot,
        right_pos=right_pos,
        right_rot=right_rot,
        left_gripper=comps["left_gripper"],
        right_gripper=comps["right_gripper"],
    )


def convert_world_action_to_frame(action_world: torch.Tensor, world_T_frame: torch.Tensor) -> torch.Tensor:
    world_T_world = eye_transform(action_world.shape[0], 1, dtype=action_world.dtype, device=action_world.device)
    return convert_pose_action_between_frames(action_world, world_T_world, world_T_frame)


def convert_frame_action_to_world(action_frame: torch.Tensor, world_T_frame: torch.Tensor) -> torch.Tensor:
    world_T_world = eye_transform(action_frame.shape[0], 1, dtype=action_frame.dtype, device=action_frame.device)
    return convert_pose_action_between_frames(action_frame, world_T_frame, world_T_world)


def convert_base_pose_action_to_rel_trans(
    base_action: torch.Tensor,
    left_base_pos_ref: torch.Tensor,
    right_base_pos_ref: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            base_action[..., LEFT_POS_SLICE] - left_base_pos_ref,
            base_action[..., LEFT_ROT_SLICE],
            base_action[..., RIGHT_POS_SLICE] - right_base_pos_ref,
            base_action[..., RIGHT_ROT_SLICE],
            base_action[..., 18:],
        ],
        dim=-1,
    )


def convert_base_rel_trans_to_pose_action(
    rel_trans_action: torch.Tensor,
    left_base_pos_ref: torch.Tensor,
    right_base_pos_ref: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            rel_trans_action[..., LEFT_POS_SLICE] + left_base_pos_ref,
            rel_trans_action[..., LEFT_ROT_SLICE],
            rel_trans_action[..., RIGHT_POS_SLICE] + right_base_pos_ref,
            rel_trans_action[..., RIGHT_ROT_SLICE],
            rel_trans_action[..., 18:],
        ],
        dim=-1,
    )


def _expand_ref_to_horizon(ref: torch.Tensor, horizon: int) -> torch.Tensor:
    if ref.shape[1] == horizon:
        return ref
    if ref.shape[1] != 1:
        raise ValueError(f"Expected reference horizon 1 or {horizon}, got shape {ref.shape}.")
    return ref.expand(-1, horizon, -1)


def convert_base_pose_action_to_rel_traj(
    base_action: torch.Tensor,
    left_base_pos_ref: torch.Tensor,
    left_base_quat_ref: torch.Tensor,
    right_base_pos_ref: torch.Tensor,
    right_base_quat_ref: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    horizon = base_action.shape[1]
    left_base_pos_ref = _expand_ref_to_horizon(left_base_pos_ref, horizon)
    left_base_quat_ref = _expand_ref_to_horizon(left_base_quat_ref, horizon)
    right_base_pos_ref = _expand_ref_to_horizon(right_base_pos_ref, horizon)
    right_base_quat_ref = _expand_ref_to_horizon(right_base_quat_ref, horizon)
    left_ref_rot = quat_to_mat.forward(left_base_quat_ref)
    right_ref_rot = quat_to_mat.forward(right_base_quat_ref)

    comps = split_action_pose_components(base_action)
    left_rot_rel = torch.matmul(left_ref_rot.transpose(-1, -2), comps["left_rot"])
    right_rot_rel = torch.matmul(right_ref_rot.transpose(-1, -2), comps["right_rot"])
    left_pos_rel = torch.einsum(
        "...ij,...j->...i",
        left_ref_rot.transpose(-1, -2),
        comps["left_pos"] - left_base_pos_ref,
    )
    right_pos_rel = torch.einsum(
        "...ij,...j->...i",
        right_ref_rot.transpose(-1, -2),
        comps["right_pos"] - right_base_pos_ref,
    )
    return assemble_action_pose_components(
        left_pos=left_pos_rel,
        left_rot=left_rot_rel,
        right_pos=right_pos_rel,
        right_rot=right_rot_rel,
        left_gripper=comps["left_gripper"],
        right_gripper=comps["right_gripper"],
    )


def convert_rel_traj_to_base_pose_action(
    rel_traj_action: torch.Tensor,
    left_base_pos_ref: torch.Tensor,
    left_base_quat_ref: torch.Tensor,
    right_base_pos_ref: torch.Tensor,
    right_base_quat_ref: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    horizon = rel_traj_action.shape[1]
    left_base_pos_ref = _expand_ref_to_horizon(left_base_pos_ref, horizon)
    left_base_quat_ref = _expand_ref_to_horizon(left_base_quat_ref, horizon)
    right_base_pos_ref = _expand_ref_to_horizon(right_base_pos_ref, horizon)
    right_base_quat_ref = _expand_ref_to_horizon(right_base_quat_ref, horizon)
    left_ref_rot = quat_to_mat.forward(left_base_quat_ref)
    right_ref_rot = quat_to_mat.forward(right_base_quat_ref)

    comps = split_action_pose_components(rel_traj_action)
    left_rot_base = torch.matmul(left_ref_rot, comps["left_rot"])
    right_rot_base = torch.matmul(right_ref_rot, comps["right_rot"])
    left_pos_base = torch.einsum("...ij,...j->...i", left_ref_rot, comps["left_pos"]) + left_base_pos_ref
    right_pos_base = torch.einsum("...ij,...j->...i", right_ref_rot, comps["right_pos"]) + right_base_pos_ref
    return assemble_action_pose_components(
        left_pos=left_pos_base,
        left_rot=left_rot_base,
        right_pos=right_pos_base,
        right_rot=right_rot_base,
        left_gripper=comps["left_gripper"],
        right_gripper=comps["right_gripper"],
    )


def convert_base_pose_action_to_rel_traj_as_vectors(
    base_action: torch.Tensor,
    left_base_pos_ref: torch.Tensor,
    left_base_quat_ref: torch.Tensor,
    right_base_pos_ref: torch.Tensor,
    right_base_quat_ref: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    """Like convert_base_pose_action_to_rel_traj but treats rot6d as raw vectors.

    Use this for noisy DDIM samples whose rot6d columns are not valid rotation
    columns.  Position gets the full rigid transform (subtract ref, rotate by
    R_ref^T) while the two rot6d 3-vectors per arm are rotated independently
    without projecting through Gram-Schmidt.
    """
    horizon = base_action.shape[1]
    left_base_pos_ref = _expand_ref_to_horizon(left_base_pos_ref, horizon)
    left_base_quat_ref = _expand_ref_to_horizon(left_base_quat_ref, horizon)
    right_base_pos_ref = _expand_ref_to_horizon(right_base_pos_ref, horizon)
    right_base_quat_ref = _expand_ref_to_horizon(right_base_quat_ref, horizon)
    left_ref_rot_t = quat_to_mat.forward(left_base_quat_ref).transpose(-1, -2)
    right_ref_rot_t = quat_to_mat.forward(right_base_quat_ref).transpose(-1, -2)

    out = base_action.clone()
    # Position: subtract ref then rotate by R_ref^T
    out[..., LEFT_POS_SLICE] = torch.einsum(
        "...ij,...j->...i",
        left_ref_rot_t,
        base_action[..., LEFT_POS_SLICE] - left_base_pos_ref,
    )
    out[..., RIGHT_POS_SLICE] = torch.einsum(
        "...ij,...j->...i",
        right_ref_rot_t,
        base_action[..., RIGHT_POS_SLICE] - right_base_pos_ref,
    )
    # Rot6d: rotate each 3-vector column independently by R_ref^T
    left_rot_vecs = base_action[..., LEFT_ROT_SLICE].unflatten(-1, (2, 3))
    right_rot_vecs = base_action[..., RIGHT_ROT_SLICE].unflatten(-1, (2, 3))
    out[..., LEFT_ROT_SLICE] = torch.einsum(
        "...ij,...kj->...ki", left_ref_rot_t, left_rot_vecs
    ).flatten(-2, -1)
    out[..., RIGHT_ROT_SLICE] = torch.einsum(
        "...ij,...kj->...ki", right_ref_rot_t, right_rot_vecs
    ).flatten(-2, -1)
    return out


def convert_rel_traj_to_base_pose_action_as_vectors(
    rel_traj_action: torch.Tensor,
    left_base_pos_ref: torch.Tensor,
    left_base_quat_ref: torch.Tensor,
    right_base_pos_ref: torch.Tensor,
    right_base_quat_ref: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    """Like convert_rel_traj_to_base_pose_action but treats rot6d as raw vectors.

    Use this for noisy DDIM samples whose rot6d columns are not valid rotation
    columns.  Position gets the full rigid transform (rotate by R_ref, add ref)
    while the two rot6d 3-vectors per arm are rotated independently without
    projecting through Gram-Schmidt.
    """
    horizon = rel_traj_action.shape[1]
    left_base_pos_ref = _expand_ref_to_horizon(left_base_pos_ref, horizon)
    left_base_quat_ref = _expand_ref_to_horizon(left_base_quat_ref, horizon)
    right_base_pos_ref = _expand_ref_to_horizon(right_base_pos_ref, horizon)
    right_base_quat_ref = _expand_ref_to_horizon(right_base_quat_ref, horizon)
    left_ref_rot = quat_to_mat.forward(left_base_quat_ref)
    right_ref_rot = quat_to_mat.forward(right_base_quat_ref)

    out = rel_traj_action.clone()
    # Position: rotate by R_ref then add ref
    out[..., LEFT_POS_SLICE] = (
        torch.einsum("...ij,...j->...i", left_ref_rot, rel_traj_action[..., LEFT_POS_SLICE])
        + left_base_pos_ref
    )
    out[..., RIGHT_POS_SLICE] = (
        torch.einsum("...ij,...j->...i", right_ref_rot, rel_traj_action[..., RIGHT_POS_SLICE])
        + right_base_pos_ref
    )
    # Rot6d: rotate each 3-vector column independently by R_ref
    left_rot_vecs = rel_traj_action[..., LEFT_ROT_SLICE].unflatten(-1, (2, 3))
    right_rot_vecs = rel_traj_action[..., RIGHT_ROT_SLICE].unflatten(-1, (2, 3))
    out[..., LEFT_ROT_SLICE] = torch.einsum(
        "...ij,...kj->...ki", left_ref_rot, left_rot_vecs
    ).flatten(-2, -1)
    out[..., RIGHT_ROT_SLICE] = torch.einsum(
        "...ij,...kj->...ki", right_ref_rot, right_rot_vecs
    ).flatten(-2, -1)
    return out


def convert_base_delta_to_rel_traj(
    base_delta: torch.Tensor,
    left_base_quat_ref: torch.Tensor,
    right_base_quat_ref: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    horizon = base_delta.shape[1]
    left_base_quat_ref = _expand_ref_to_horizon(left_base_quat_ref, horizon)
    right_base_quat_ref = _expand_ref_to_horizon(right_base_quat_ref, horizon)
    left_ref_rot_t = quat_to_mat.forward(left_base_quat_ref).transpose(-1, -2)
    right_ref_rot_t = quat_to_mat.forward(right_base_quat_ref).transpose(-1, -2)
    out = base_delta.clone()
    left_vecs = out[..., :9].unflatten(-1, (3, 3))
    right_vecs = out[..., 9:18].unflatten(-1, (3, 3))
    out[..., :9] = torch.einsum("...ij,...kj->...ki", left_ref_rot_t, left_vecs).flatten(-2, -1)
    out[..., 9:18] = torch.einsum("...ij,...kj->...ki", right_ref_rot_t, right_vecs).flatten(-2, -1)
    return out


def convert_rel_traj_delta_to_base(
    rel_traj_delta: torch.Tensor,
    left_base_quat_ref: torch.Tensor,
    right_base_quat_ref: torch.Tensor,
    quat_to_mat: RotationTransformer,
) -> torch.Tensor:
    horizon = rel_traj_delta.shape[1]
    left_base_quat_ref = _expand_ref_to_horizon(left_base_quat_ref, horizon)
    right_base_quat_ref = _expand_ref_to_horizon(right_base_quat_ref, horizon)
    left_ref_rot = quat_to_mat.forward(left_base_quat_ref)
    right_ref_rot = quat_to_mat.forward(right_base_quat_ref)
    out = rel_traj_delta.clone()
    left_vecs = out[..., :9].unflatten(-1, (3, 3))
    right_vecs = out[..., 9:18].unflatten(-1, (3, 3))
    out[..., :9] = torch.einsum("...ij,...kj->...ki", left_ref_rot, left_vecs).flatten(-2, -1)
    out[..., 9:18] = torch.einsum("...ij,...kj->...ki", right_ref_rot, right_vecs).flatten(-2, -1)
    return out


def transform_obs_pose_dict_to_frame(
    obs_dict: Dict[str, torch.Tensor],
    world_T_frame: torch.Tensor,
    quat_to_mat: RotationTransformer,
    mat_to_quat: RotationTransformer,
    pos_keys: Iterable[str] = POSE_POSITION_KEYS,
    quat_keys: Iterable[str] = POSE_QUAT_KEYS,
) -> Dict[str, torch.Tensor]:
    out = dict(obs_dict)
    frame_T_world = torch.linalg.inv(world_T_frame)
    for key in pos_keys:
        if key not in out:
            continue
        out[key] = transform_points(frame_T_world, out[key])
    for key in quat_keys:
        if key not in out:
            continue
        rot_world = quat_to_mat.forward(out[key])
        rot_frame = torch.matmul(frame_T_world[..., :3, :3], rot_world)
        out[key] = mat_to_quat.forward(rot_frame)
    return out


def normalize_like_delta(delta_real: torch.Tensor, normalizer) -> torch.Tensor:
    scale = normalizer.params_dict["scale"].to(device=delta_real.device, dtype=delta_real.dtype)
    view_shape = (1,) * (delta_real.ndim - 1) + (delta_real.shape[-1],)
    return delta_real * scale.view(view_shape)


def unnormalize_like_delta(delta_norm: torch.Tensor, normalizer) -> torch.Tensor:
    scale = normalizer.params_dict["scale"].to(device=delta_norm.device, dtype=delta_norm.dtype)
    view_shape = (1,) * (delta_norm.ndim - 1) + (delta_norm.shape[-1],)
    return delta_norm / scale.view(view_shape)
