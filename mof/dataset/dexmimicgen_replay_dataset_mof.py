"""Dataset for DexMimicGen HDF5 data, compatible with the Mixture of Frames pipeline.

Maps dexmimicgen dual-arm observations and actions into the same format as
BigymRBY1ReplayDatasetMoF so the existing Mixture of Frames policy can train
without modification.

IMPORTANT: The input HDF5 must have absolute actions (not delta).
Run the conversion script first:
    python -m mof.scripts.dexmimicgen_dataset_conversion \
        -i data/dexmimicgen/two_arm_threading.hdf5 \
        -o data/dexmimicgen/two_arm_threading_abs.hdf5 -e eval_dir -n 8

Arm mapping (robosuite convention for two-arm tasks):
    robot0 = right arm,  robot1 = left arm

Quaternion convention:
    robosuite stores XYZW (scipy), pytorch3d expects WXYZ.
    We convert on load.
"""

from typing import Dict, List
import copy
import os
import shutil

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from mof.common.replay_buffer import ReplayBuffer
from mof.common.sampler import SequenceSampler, get_val_mask
from mof.dataset.bigym_rby1_replay_dataset_mof import (
    BigymRBY1ReplayDatasetMoF,
)
from mof.model.common.rotation_transformer import RotationTransformer
from mof.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs(verbose=False)

# Robosuite XYZW → pytorch3d WXYZ reorder indices
_XYZW_TO_WXYZ = [3, 0, 1, 2]


def _quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion array from XYZW (robosuite/scipy) to WXYZ (pytorch3d)."""
    return quat[..., _XYZW_TO_WXYZ].copy()


def _convert_dexmimicgen_hdf5_to_replay(
    store,
    shape_meta: dict,
    dataset_path: str,
    n_demo: int = 100,
    n_workers: int = None,
):
    """Load a DexMimicGen HDF5 file into a ReplayBuffer with BigYM-compatible keys.

    The replay buffer will contain:
      RGB:     head_image, left_wrist_image, right_wrist_image  (H, W, C uint8)
      Lowdim:  left_ee_pos, left_ee_quat, right_ee_pos, right_ee_quat,
               proprioception_grippers, base_pos, base_quat
      Action:  20-dim absolute [left_pos(3), left_rot6d(6), right_pos(3),
               right_rot6d(6), left_gripper(1), right_gripper(1)]
    """
    import multiprocessing
    import concurrent.futures

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    aa_to_rot6d = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

    # Observation key mapping: dexmimicgen_key -> (replay_buffer_key, transform_fn)
    # robot0 = right arm, robot1 = left arm
    RGB_MAP = {
        "obs/agentview_image": "head_image",
        "obs/robot1_eye_in_hand_image": "left_wrist_image",
        "obs/robot0_eye_in_hand_image": "right_wrist_image",
    }
    LOWDIM_MAP = {
        "obs/robot1_eef_pos": ("left_ee_pos", None),
        "obs/robot1_eef_quat": ("left_ee_quat", _quat_xyzw_to_wxyz),
        "obs/robot0_eef_pos": ("right_ee_pos", None),
        "obs/robot0_eef_quat": ("right_ee_quat", _quat_xyzw_to_wxyz),
    }

    # Parse shape_meta to know which keys to store
    rgb_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type_ = attr.get("type", "low_dim")
        if type_ == "rgb":
            rgb_keys.append(key)
        elif type_ == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path, "r") as f:
        demos_group = f["data"]
        n_available = len([k for k in demos_group.keys() if k.startswith("demo_")])
        n_demo = min(n_demo, n_available)
        print("Loading %d / %d demos from %s" % (n_demo, n_available, dataset_path))

        # First pass: count steps and build episode_ends
        episode_ends = []
        prev_end = 0
        for i in range(n_demo):
            demo = demos_group["demo_%d" % i]
            ep_len = demo["actions"].shape[0]
            prev_end += ep_len
            episode_ends.append(prev_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]

        meta_group.array(
            "episode_ends", episode_ends, dtype=np.int64,
            compressor=None, overwrite=True,
        )

        # ---- Load low-dim observations ----
        lowdim_arrays = {}  # replay_buffer_key -> list of arrays
        for replay_key in lowdim_keys:
            lowdim_arrays[replay_key] = []

        action_list = []

        for i in tqdm(range(n_demo), desc="Loading lowdim + actions"):
            demo = demos_group["demo_%d" % i]

            # Map lowdim observations
            for src_key, (dst_key, transform) in LOWDIM_MAP.items():
                arr = demo[src_key][:].astype(np.float32)
                if transform is not None:
                    arr = transform(arr)
                lowdim_arrays[dst_key].append(arr)

            # Proprioception grippers: take the leading per_arm channels of each arm's
            # gripper qpos. per_arm is derived from shape_meta:
            #   parallel-jaw → shape [2]   → per_arm=1 (first finger joint)
            #   dex hand     → shape [24]  → per_arm=12 (all hand joints)
            # robot1=left, robot0=right
            propr_total = shape_meta["obs"]["proprioception_grippers"]["shape"][0]
            assert propr_total % 2 == 0, (
                "proprioception_grippers shape must be even, got %d" % propr_total
            )
            per_arm = propr_total // 2
            left_grip = demo["obs/robot1_gripper_qpos"][:, 0:per_arm].astype(np.float32)
            right_grip = demo["obs/robot0_gripper_qpos"][:, 0:per_arm].astype(np.float32)
            lowdim_arrays["proprioception_grippers"].append(
                np.concatenate([left_grip, right_grip], axis=-1)
            )

            # Fixed base (no mobile base in dexmimicgen)
            ep_len = demo["actions"].shape[0]
            lowdim_arrays["base_pos"].append(
                np.zeros((ep_len, 3), dtype=np.float32)
            )
            lowdim_arrays["base_quat"].append(
                np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (ep_len, 1))
            )

            # Load absolute actions from HDF5 (must be pre-converted)
            # HDF5 format: [robot0(7), robot1(7)] = [right(7), left(7)]
            # Each arm: [pos(3), ori_axis_angle(3), gripper(1)]
            action_list.append(demo["actions"][:].astype(np.float32))

        # Store lowdim arrays in replay buffer
        for key in lowdim_keys:
            arr = np.concatenate(lowdim_arrays[key], axis=0)
            expected_shape = (n_steps,) + tuple(shape_meta["obs"][key]["shape"])
            assert arr.shape == expected_shape, (
                "Shape mismatch for %s: got %s, expected %s" % (key, arr.shape, expected_shape)
            )
            data_group.array(
                name=key, data=arr, shape=arr.shape,
                chunks=arr.shape, compressor=None, dtype=arr.dtype,
            )

        # ---- Convert absolute actions to Mixture of Frames format ----
        # HDF5 layout: [robot0(arm_dim), robot1(arm_dim)] where robot0=right, robot1=left.
        # Each arm: [pos(3), ori_axis_angle(3), gripper(grip_dim)]; arm_dim = 6 + grip_dim.
        #   parallel-jaw: arm_dim=7, grip_dim=1, mof action_dim=20
        #   dex hand:     arm_dim=12, grip_dim=6, mof action_dim=30
        # Target: [left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6),
        #          left_gripper(grip_dim), right_gripper(grip_dim)]
        raw_actions = np.concatenate(action_list, axis=0)  # (N, 2*arm_dim)
        arm_dim = raw_actions.shape[-1] // 2
        assert raw_actions.shape == (n_steps, 2 * arm_dim) and arm_dim in (7, 12), (
            "Expected per-arm width 7 (parallel) or 12 (dex), got action shape %s. "
            "Run dexmimicgen_dataset_conversion.py first." % (raw_actions.shape,)
        )
        grip_dim = arm_dim - 6
        target_action_dim = 18 + 2 * grip_dim
        expected_action_dim = shape_meta["action"]["shape"][0]
        assert expected_action_dim == target_action_dim, (
            "shape_meta action shape %d does not match HDF5-derived action_dim %d "
            "(arm_dim=%d, grip_dim=%d)"
            % (expected_action_dim, target_action_dim, arm_dim, grip_dim)
        )

        # Split into per-arm components: robot0=right, robot1=left
        stacked = raw_actions.reshape(n_steps, 2, arm_dim)
        right_pos = stacked[:, 0, :3]              # robot0 = right
        right_aa = stacked[:, 0, 3:6]
        right_gripper = stacked[:, 0, 6:arm_dim]
        left_pos = stacked[:, 1, :3]               # robot1 = left
        left_aa = stacked[:, 1, 3:6]
        left_gripper = stacked[:, 1, 6:arm_dim]

        # Convert axis-angle to 6D rotation (pytorch3d row convention)
        left_rot6d = aa_to_rot6d.forward(left_aa)
        right_rot6d = aa_to_rot6d.forward(right_aa)
        if isinstance(left_rot6d, torch.Tensor):
            left_rot6d = left_rot6d.numpy()
        if isinstance(right_rot6d, torch.Tensor):
            right_rot6d = right_rot6d.numpy()

        actions = np.concatenate([
            left_pos,           # 0:3
            left_rot6d,         # 3:9
            right_pos,          # 9:12
            right_rot6d,        # 12:18
            left_gripper,       # 18:18+grip_dim
            right_gripper,      # 18+grip_dim:18+2*grip_dim
        ], axis=-1).astype(np.float32)

        assert actions.shape == (n_steps, target_action_dim), (
            "Action shape mismatch: got %s, expected (%d, %d)"
            % (actions.shape, n_steps, target_action_dim)
        )
        data_group.array(
            name="action", data=actions, shape=actions.shape,
            chunks=actions.shape, compressor=None, dtype=actions.dtype,
        )

        # ---- Load RGB images (batched per episode) ----
        def img_write_single(zarr_arr, zarr_idx, np_img):
            """Write a single numpy image to zarr (with JPEG2K compression)."""
            try:
                zarr_arr[zarr_idx] = np_img
                _ = zarr_arr[zarr_idx]  # verify decode
                return True
            except Exception:
                return False

        max_inflight = n_workers * 5
        with tqdm(total=n_steps * len(rgb_keys), desc="Loading images", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for replay_key in rgb_keys:
                    # Find the source HDF5 key
                    src_key = None
                    for hdf5_key, rk in RGB_MAP.items():
                        if rk == replay_key:
                            src_key = hdf5_key
                            break
                    if src_key is None:
                        raise ValueError("No HDF5 source for RGB key: %s" % replay_key)

                    shape = tuple(shape_meta["obs"][replay_key]["shape"])
                    c, h, w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=replay_key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8,
                    )
                    for ep_idx in range(n_demo):
                        demo = demos_group["demo_%d" % ep_idx]
                        # Batch-read entire episode from HDF5 (one bulk read)
                        episode_imgs = demo[src_key][:]  # (T, H, W, C) numpy
                        for frame_idx in range(episode_imgs.shape[0]):
                            if len(futures) >= max_inflight:
                                completed, futures = concurrent.futures.wait(
                                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for ft in completed:
                                    if not ft.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[ep_idx] + frame_idx
                            futures.add(
                                executor.submit(
                                    img_write_single, img_arr, zarr_idx,
                                    episode_imgs[frame_idx],
                                )
                            )
                    completed, futures = concurrent.futures.wait(futures)
                    for ft in completed:
                        if not ft.result():
                            raise RuntimeError("Failed to encode image!")
                    pbar.update(len(completed))

    return ReplayBuffer(root)


class DexMimicGenReplayDatasetMoF(BigymRBY1ReplayDatasetMoF):
    """Mixture of Frames dataset for DexMimicGen HDF5 data.

    Loads dexmimicgen dual-arm HDF5 data, maps observations and actions
    to the BigYM RBY1 convention, then reuses all frame precomputation
    and normalization logic from the parent class.
    """

    def __init__(
        self,
        shape_meta: dict,
        demos_dir: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        n_demo=None,
        normalize_rgb=True,
    ):
        # Resolve path - demos_dir points to base, HDF5 is at {demos_dir}.hdf5
        if not os.path.isabs(demos_dir):
            try:
                from hydra.utils import to_absolute_path as hydra_to_absolute_path
                demos_dir = hydra_to_absolute_path(demos_dir)
            except Exception:
                demos_dir = os.path.abspath(demos_dir)

        dataset_path = demos_dir + ".hdf5"
        if not os.path.exists(dataset_path):
            # Maybe demos_dir itself is the HDF5 file
            if os.path.exists(demos_dir) and demos_dir.endswith(".hdf5"):
                dataset_path = demos_dir
            else:
                raise FileNotFoundError(
                    "Cannot find HDF5 file. Tried: %s" % dataset_path
                )

        self.n_demo = n_demo
        self.normalize_rgb = normalize_rgb

        # ---- Create replay buffer from HDF5 ----
        replay_buffer = None

        # Use the directory containing the HDF5 for cache files
        cache_dir = os.path.dirname(dataset_path)
        cache_base = os.path.splitext(os.path.basename(dataset_path))[0]

        if use_cache:
            cache_zarr_path = os.path.join(
                cache_dir, "cache_dexmimicgen_%s_%s.zarr.zip" % (cache_base, n_demo)
            )
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_dexmimicgen_hdf5_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            n_demo=n_demo,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_dexmimicgen_hdf5_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                n_demo=n_demo,
            )

        # ---- From here, identical to parent class init ----
        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type_ = attr.get("type", "low_dim")
            if type_ == "rgb":
                rgb_keys.append(key)
            elif type_ == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
            for key in ("base_pos", "base_quat", "left_ee_pos", "left_ee_quat",
                        "right_ee_pos", "right_ee_quat"):
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.key_first_k = key_first_k
        self.quat_to_mat = RotationTransformer(from_rep="quaternion", to_rep="matrix")
        self.mat_to_quat = RotationTransformer(from_rep="matrix", to_rep="quaternion")
        self.base_frame_lowdim = self._precompute_base_frame_lowdim()
        self.left_frame_lowdim = self._precompute_side_frame_lowdim(frame_name="left")
        self.right_frame_lowdim = self._precompute_side_frame_lowdim(frame_name="right")
        self.actions = self._precompute_world_action_chunks()
        self.base_actions = self._precompute_frame_action_chunks(frame_name="base")
        self.left_actions = self._precompute_frame_action_chunks(frame_name="left")
        self.right_actions = self._precompute_frame_action_chunks(frame_name="right")
        self.base_rel_trans_actions = self._precompute_base_rel_trans_action_chunks()
        self.rel_traj_actions = self._precompute_rel_traj_action_chunks()
        self._precompute_frame_obs_chunks()
        self._precompute_world_ref_chunks()

    def get_normalizer(self, **kwargs):
        """Build range normalizers.

        Parallel-jaw gripper (grip_dim=1, in [-1, 1])     → identity normalization.
        Dex hand gripper (grip_dim=6, joint-space qpos)   → range normalization
          from actual stats; otherwise diffusion clip_sample=True clamps valid joint
          commands above 1.0 (dex qpos goes up to π/2 ≈ 1.57).
        """
        from mof.common.normalize_util import (
            array_to_stats,
            robosuite_dual_arm_action_normalizer_from_stat,
            robosuite_dual_arm_action_normalizer_dex_from_stat,
            get_identity_normalizer_from_stat,
            get_image_identity_normalizer,
            get_image_range_normalizer,
            get_range_normalizer_from_stat,
        )
        from mof.model.common.normalizer import LinearNormalizer

        normalizer = LinearNormalizer()

        action_dim = self.replay_buffer["action"].shape[-1]
        grip_dim = (action_dim - 18) // 2
        action_normalizer_fn = (
            robosuite_dual_arm_action_normalizer_from_stat
            if grip_dim == 1
            else robosuite_dual_arm_action_normalizer_dex_from_stat
        )

        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = action_normalizer_fn(action_stat)

        expert_data = {
            "base": ("base_action", self.base_actions),
            "left": ("left_action", self.left_actions),
            "right": ("right_action", self.right_actions),
            "base_rel_trans": ("base_rel_trans_action", self.base_rel_trans_actions),
            "rel_traj": ("rel_traj_action", self.rel_traj_actions),
        }
        for name, (norm_key, data) in expert_data.items():
            stat = array_to_stats(data.reshape(-1, action_dim))
            normalizer[norm_key] = action_normalizer_fn(stat)

        # Lowdim obs normalizers (per frame)
        for frame_name, frame_lowdim in (
            ("base", self.base_frame_lowdim),
            ("left", self.left_frame_lowdim),
            ("right", self.right_frame_lowdim),
        ):
            for key in self.lowdim_keys:
                if key.endswith("pos"):
                    stat = array_to_stats(frame_lowdim[key])
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif key.endswith("quat"):
                    stat = array_to_stats(frame_lowdim[key])
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                else:
                    stat = array_to_stats(frame_lowdim[key])
                    this_normalizer = get_range_normalizer_from_stat(stat)
                normalizer[self._frame_key(frame_name, key)] = this_normalizer

        # RGB normalizers
        for key in self.rgb_keys:
            normalizer[key] = (
                get_image_range_normalizer() if self.normalize_rgb
                else get_image_identity_normalizer()
            )
        return normalizer
