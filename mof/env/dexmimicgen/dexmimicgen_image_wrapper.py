"""Gym wrapper for dexmimicgen robosuite environments.

Maps robosuite observation keys to the BigYM-style naming convention
used by the Mixture of Frames pipeline.

Arm mapping: robot0 = right, robot1 = left.
Quaternion convention: robosuite XYZW → pytorch3d WXYZ.
"""

import numpy as np
import gym
from gym import spaces
from robomimic.envs.env_robosuite import EnvRobosuite

# Robosuite XYZW → pytorch3d WXYZ
_XYZW_TO_WXYZ = [3, 0, 1, 2]

# Observation key mapping: robosuite_key -> our_key
_OBS_KEY_MAP = {
    "agentview_image": "head_image",
    "robot1_eye_in_hand_image": "left_wrist_image",
    "robot0_eye_in_hand_image": "right_wrist_image",
    "robot1_eef_pos": "left_ee_pos",
    "robot1_eef_quat": "left_ee_quat",
    "robot0_eef_pos": "right_ee_pos",
    "robot0_eef_quat": "right_ee_quat",
}

# Reverse mapping for lookup
_OUR_KEY_TO_ROBOSUITE = {v: k for k, v in _OBS_KEY_MAP.items()}


class DexMimicGenImageWrapper(gym.Env):
    """Wraps a robosuite dexmimicgen env, remapping observations to Mixture of Frames keys."""

    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state=None,
        render_obs_key="head_image",
    ):
        self.env = env
        self.init_state = init_state
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self._seed = None
        self.seed_state_map = {}

        # Map render_obs_key to robosuite key
        self._render_robosuite_key = _OUR_KEY_TO_ROBOSUITE.get(
            render_obs_key, render_obs_key
        )

        # Setup spaces
        action_shape = shape_meta["action"]["shape"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=action_shape, dtype=np.float32
        )

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_val, max_val = -1, 1
            if key.endswith("image"):
                min_val, max_val = 0, 1
            observation_space[key] = spaces.Box(
                low=min_val, high=max_val, shape=shape, dtype=np.float32
            )
        self.observation_space = observation_space

    def _map_observation(self, raw_obs):
        """Map robosuite observation dict to our naming convention."""
        obs = {}
        for our_key in self.observation_space.keys():
            if our_key == "proprioception_grippers":
                # Take the leading per_arm channels of each arm's gripper qpos.
                # Width derived from shape_meta:
                #   parallel-jaw → [2]   → per_arm=1 (first finger joint)
                #   dex hand     → [24]  → per_arm=12 (all hand joints)
                propr_total = self.observation_space[our_key].shape[0]
                per_arm = propr_total // 2
                left_grip = raw_obs["robot1_gripper_qpos"][0:per_arm]
                right_grip = raw_obs["robot0_gripper_qpos"][0:per_arm]
                obs[our_key] = np.concatenate([left_grip, right_grip]).astype(
                    np.float32
                )
            elif our_key == "base_pos":
                obs[our_key] = np.zeros(3, dtype=np.float32)
            elif our_key == "base_quat":
                obs[our_key] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif our_key in _OUR_KEY_TO_ROBOSUITE:
                robosuite_key = _OUR_KEY_TO_ROBOSUITE[our_key]
                val = raw_obs[robosuite_key]
                # Convert quaternion convention
                if our_key.endswith("_quat"):
                    val = val[_XYZW_TO_WXYZ].copy()
                obs[our_key] = val.astype(np.float32)
            else:
                raise KeyError(
                    "No robosuite mapping for obs key: %s" % our_key
                )

        # Cache render image
        if self._render_robosuite_key in raw_obs:
            self.render_cache = raw_obs[self._render_robosuite_key]

        return obs

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        return self._map_observation(raw_obs)

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img
