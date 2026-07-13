"""Env runner for dexmimicgen tasks with Mixture of Frames policy.

Handles:
- robosuite 1.5.x composite controller API (input_type instead of control_delta)
- Mixture of Frames 20D action → robosuite 14D absolute action conversion
- Observation key remapping via DexMimicGenImageWrapper
"""

import os
import collections
import math
import pathlib

import dill
import h5py
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from mof.common.pytorch_util import dict_apply
from mof.env.dexmimicgen.dexmimicgen_image_wrapper import DexMimicGenImageWrapper
from mof.env_runner.base_image_runner import BaseImageRunner
from mof.gym_util.async_vector_env import AsyncVectorEnv
from mof.gym_util.async_vector_env_ddp import AsyncVectorEnvDDP
from mof.gym_util.multistep_wrapper import MultiStepWrapper
from mof.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)
from mof.model.common.rotation_transformer import RotationTransformer
from mof.policy.base_image_policy import BaseImagePolicy

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


def _create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    # Use robosuite obs keys for modality mapping
    robosuite_rgb = ["agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"]
    robosuite_lowdim = [
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
        "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
    ]
    modality_mapping["rgb"] = robosuite_rgb
    modality_mapping["low_dim"] = robosuite_lowdim
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class DexMimicGenImageRunner(BaseImageRunner):
    """Evaluation runner for dexmimicgen tasks with Mixture of Frames policy."""

    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=0,
        n_train_vis=0,
        train_start_idx=0,
        n_test=50,
        n_test_vis=5,
        test_start_seed=100000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="head_image",
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        disable_tqdm=False,
        use_ddp_async_env=False,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta["env_kwargs"]["use_object_obs"] = False

        # Set absolute control mode (robosuite 1.5.x API)
        ctrl_cfg = env_meta["env_kwargs"]["controller_configs"]
        for part_name, part_cfg in ctrl_cfg.get("body_parts", {}).items():
            if "input_type" in part_cfg:
                part_cfg["input_type"] = "absolute"

        # Extract robot base poses for world→robot-base-frame conversion at inference.
        # The policy predicts world-frame poses, but the controller expects robot-base-frame.
        tmp_env = _create_env(env_meta=env_meta, shape_meta=shape_meta, enable_render=False)
        tmp_env.reset()
        robot_base_pos = []
        robot_base_ori = []
        robot_base_ori_inv = []
        for robot in tmp_env.env.robots:
            robot_base_pos.append(robot.base_pos.copy())
            robot_base_ori.append(robot.base_ori.copy())
            robot_base_ori_inv.append(robot.base_ori.T.copy())  # R^T = R^{-1} for rotation
        del tmp_env

        self.robot_base_pos = robot_base_pos  # list of (3,) per robot
        self.robot_base_ori_inv = robot_base_ori_inv  # list of (3,3) per robot

        # Rotation transformer for 6D → axis-angle conversion
        self.rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        def env_fn():
            robomimic_env = _create_env(env_meta=env_meta, shape_meta=shape_meta)
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    DexMimicGenImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        def dummy_env_fn():
            robomimic_env = _create_env(
                env_meta=env_meta, shape_meta=shape_meta, enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    DexMimicGenImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = []
        env_prefixs = []
        env_init_fn_dills = []

        # train episodes (from dataset initial states)
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f["data/demo_%d/states" % train_idx][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", wv.util.generate_id() + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, DexMimicGenImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train/")
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test episodes (random seeds)
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, DexMimicGenImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        async_env_cls = AsyncVectorEnvDDP if use_ddp_async_env else AsyncVectorEnv
        env = async_env_cls(env_fns, dummy_env_fn=dummy_env_fn)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.disable_tqdm = bool(disable_tqdm)
        self.use_ddp_async_env = bool(use_ddp_async_env)
        self.max_rewards = {}
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0

    def _world_to_robot_base(self, pos_world, ori_aa_world, robot_idx):
        """Transform position and orientation from world frame to robot base frame."""
        from scipy.spatial.transform import Rotation

        R_base_inv = self.robot_base_ori_inv[robot_idx]  # (3, 3)
        t_base = self.robot_base_pos[robot_idx]  # (3,)

        # Position: R_base^{-1} @ (pos_world - t_base)
        pos_local = np.einsum("ij,...j->...i", R_base_inv, pos_world - t_base)

        # Orientation: R_base^{-1} @ R_world
        ori_shape = ori_aa_world.shape
        flat_aa = ori_aa_world.reshape(-1, 3)
        R_world = Rotation.from_rotvec(flat_aa).as_matrix()  # (N, 3, 3)
        R_local = np.einsum("ij,njk->nik", R_base_inv, R_world)
        ori_aa_local = Rotation.from_matrix(R_local).as_rotvec().reshape(ori_shape)

        return pos_local, ori_aa_local

    def undo_transform_action(self, action):
        """Convert Mixture of Frames world-frame action to robosuite robot-base-frame action.

        Per-arm gripper width is auto-detected: action_dim = 18 + 2 * grip_dim.
        Parallel-jaw uses grip_dim=1 (action_dim=20, robosuite=14);
        dex hand uses grip_dim=6 (action_dim=30, robosuite=24).

        Input (Mixture of Frames, world frame):
            [left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6),
             left_gripper(grip_dim), right_gripper(grip_dim)]

        Output (robosuite, robot base frame, robot0=right, robot1=left):
            [right_pos_local(3), right_aa_local(3), right_gripper(grip_dim),
             left_pos_local(3), left_aa_local(3), left_gripper(grip_dim)]
        """
        action_dim = action.shape[-1]
        grip_dim = (action_dim - 18) // 2

        # Parse Mixture of Frames layout
        left_pos_world = action[..., 0:3]
        left_rot6d = action[..., 3:9]
        right_pos_world = action[..., 9:12]
        right_rot6d = action[..., 12:18]
        left_gripper = action[..., 18:18 + grip_dim]
        right_gripper = action[..., 18 + grip_dim:18 + 2 * grip_dim]

        # Convert 6D rotation → axis-angle (still in world frame)
        left_aa_world = self.rotation_transformer.inverse(left_rot6d)
        right_aa_world = self.rotation_transformer.inverse(right_rot6d)

        # Transform from world frame to each robot's base frame
        # robot0 = right, robot1 = left
        right_pos_local, right_aa_local = self._world_to_robot_base(
            right_pos_world, right_aa_world, robot_idx=0
        )
        left_pos_local, left_aa_local = self._world_to_robot_base(
            left_pos_world, left_aa_world, robot_idx=1
        )

        # Assemble as robosuite format: [robot0(right)(7), robot1(left)(7)]
        env_action = np.concatenate(
            [right_pos_local, right_aa_local, right_gripper,
             left_pos_local, left_aa_local, left_gripper],
            axis=-1,
        )
        return env_action

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc="Eval %s %d/%d" % (env_name, chunk_idx + 1, n_chunks),
                leave=False,
                mininterval=self.tqdm_interval_sec,
                disable=self.disable_tqdm,
            )

            done = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                env_action = self.undo_transform_action(action)
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
        _ = env.reset()

        # Log results
        max_rewards = collections.defaultdict(list)
        log_data = {}
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + "sim_max_reward_%d" % seed] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + "sim_video_%d" % seed] = wandb.Video(video_path)

        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix + "max_score"] = self.max_rewards[prefix]

        return log_data
