from typing import Optional
import random
import numpy as np
import gym
import torch
from gym import spaces

try:
    from bigym.utils.env_health import UnstableSimulationError
except Exception:  # pragma: no cover - fallback when BiGym is unavailable
    class UnstableSimulationError(RuntimeError):
        pass


class BigymImageWrapper(gym.Env):
    def __init__(self,
        env,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key: str = 'head_image',
        normalized_action_space: bool = False,
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.rgb_obs_keys = {
            key for key, attr in shape_meta['obs'].items()
            if attr.get('type', 'low_dim') == 'rgb'
        }

        # setup spaces
        action_shape = shape_meta['action']['shape']
        if normalized_action_space:
            action_space = spaces.Box(
                low=-1,
                high=1,
                shape=tuple(action_shape),
                dtype=np.float32
            )
        else:
            # Allow original-scale actions; bounds are unbounded to avoid VectorEnv pre-checks
            action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(action_shape),
                dtype=np.float32
            )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = tuple(value['shape'])
            min_value, max_value = -1, 1
            if value.get('type', 'low_dim') == 'rgb' or key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('depth'):
                min_value, max_value = 0, 1
            elif key.endswith('voxels'):
                min_value, max_value = 0, 1
            elif key.startswith('pcd_'):
                min_value, max_value = -10, 10
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                min_value, max_value = -1, 1
            else:
                # default low-dim range
                min_value, max_value = -1, 1

            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    @property
    def control_frequency(self):
        return self.env.control_frequency

    def _standardize_rgb_obs(self, value):
        arr = np.asarray(value)
        if arr.ndim != 3:
            raise ValueError(f'RGB observation must be rank-3, got shape {arr.shape}')
        # detect channel order
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            chw = arr
            hwc = np.moveaxis(arr, 0, -1)
        else:
            hwc = arr
            chw = np.moveaxis(arr, -1, 0)
        chw = chw.astype(np.float32, copy=False) / 255.0
        render_img = hwc
        if render_img.dtype != np.uint8:
            if np.issubdtype(render_img.dtype, np.floating):
                render_img = np.clip(render_img, 0.0, 1.0)
                render_img = (render_img * 255).astype(np.uint8)
            else:
                render_img = np.clip(render_img, 0, 255).astype(np.uint8)
        return chw, render_img

    def _resolve_obs_value(self, key, raw_obs):
        if not isinstance(raw_obs, dict):
            raise KeyError(f'Observation is not a dict (key={key}).')
        if key in raw_obs:
            return raw_obs[key]
        if key in self.rgb_obs_keys:
            base = key
            if base.endswith('_image'):
                base = base[:-6]
            if base.endswith('_'):
                base = base[:-1]
            candidates = [
                f"rgb_{base}",
                base,
                f"{base}_image",
                f"{base}_rgb",
            ]
            for candidate in candidates:
                if candidate in raw_obs:
                    return raw_obs[candidate]
            if base in raw_obs and isinstance(raw_obs[base], dict):
                cam_dict = raw_obs[base]
                if 'rgb' in cam_dict:
                    return cam_dict['rgb']
            raise KeyError(f'RGB observation "{key}" not found in raw observation keys {list(raw_obs.keys())}')
        # low-dim observation missing -> raise
        raise KeyError(f'Observation "{key}" not found in raw observation keys {list(raw_obs.keys())}')

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation() if hasattr(self.env, 'get_observation') else self.last_obs

        obs = dict()
        for key in self.observation_space.keys():
            value = self._resolve_obs_value(key, raw_obs)
            attr = self.shape_meta['obs'][key]
            if attr.get('type', 'low_dim') == 'rgb':
                chw, render_candidate = self._standardize_rgb_obs(value)
                obs[key] = chw
                if key == self.render_obs_key:
                    self.render_cache = render_candidate
            else:
                obs[key] = np.asarray(value, dtype=np.float32)

        # ensure render cache populated even if render key processed earlier
        if self.render_cache is None and self.render_obs_key in obs:
            # convert back to HWC uint8 for rendering
            render_img = np.moveaxis(obs[self.render_obs_key], 0, -1)
            render_img = np.clip(render_img, 0.0, 1.0)
            self.render_cache = (render_img * 255).astype(np.uint8)
        return obs

    def seed(self, seed=None):
        # Async eval reuses persistent env subprocesses. Reset all common RNGs
        # here so env reset() starts from the same worker-local state each time.
        np.random.seed(seed=seed)
        random.seed(seed)
        self._seed = seed
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        # Bigym envs typically support gymnasium API: obs, info
        # forward stored seed to underlying env reset for determinism
        try:
            result = self.env.reset(seed=self._seed)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple):
            raw_obs, _info = result
        else:
            raw_obs = result

        self.last_obs = raw_obs
        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        try:
            result = self.env.step(action)
        except UnstableSimulationError as exc:
            # Stop this episode as failure instead of crashing rollout workers.
            raw_obs = getattr(self, "last_obs", None)
            if raw_obs is None and hasattr(self.env, "get_observation"):
                raw_obs = self.env.get_observation()
            if raw_obs is None:
                raw_obs = {
                    key: np.zeros(space.shape, dtype=np.float32)
                    for key, space in self.observation_space.items()
                }

            self.last_obs = raw_obs
            obs = self.get_observation(raw_obs)

            info = {}
            if hasattr(self.env, "get_info"):
                try:
                    info = dict(self.env.get_info())
                except Exception:
                    info = {}
            info.update(
                {
                    "unstable_simulation": True,
                    # Treat unstable simulation as episode failure.
                    "task_success": 0.0,
                    "fail": True,
                    "terminate": False,
                    "truncate": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            return obs, 0.0, True, info

        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            raw_obs, reward, done, info = result
        self.last_obs = raw_obs
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = self.render_cache
        # If already HWC uint8, return directly
        if isinstance(img, np.ndarray) and img.dtype == np.uint8:
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                return img
        # Otherwise, try to convert to HWC uint8 from CHW float
        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4):
                img = np.moveaxis(img, 0, -1)
            # scale if float
            if img.dtype != np.uint8:
                # assume 0..1 range
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return img
