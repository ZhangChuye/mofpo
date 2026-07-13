import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import importlib
from mof.gym_util.async_vector_env import AsyncVectorEnv
from mof.gym_util.async_vector_env_ddp import AsyncVectorEnvDDP
from mof.gym_util.multistep_wrapper import MultiStepWrapper
from mof.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from mof.env.bigym.bigym_image_wrapper import BigymImageWrapper
from mof.env_runner.base_image_runner import BaseImageRunner
from mof.policy.base_image_policy import BaseImagePolicy
from mof.common.pytorch_util import dict_apply


def _resolve_callable(maybe_path_or_callable):
    if callable(maybe_path_or_callable):
        return maybe_path_or_callable
    if isinstance(maybe_path_or_callable, str):
        # import from path like 'pkg.module:function'
        path = maybe_path_or_callable
        if ':' in path:
            mod_name, func_name = path.split(':', 1)
        else:
            # dotted path ending with function
            parts = path.split('.')
            mod_name, func_name = '.'.join(parts[:-1]), parts[-1]
        module = importlib.import_module(mod_name)
        return getattr(module, func_name)
    raise TypeError(f"Unsupported make_env_fn type: {type(maybe_path_or_callable)}")


def _resolution_from_shape_meta(shape_meta, render_obs_key="head_image"):
    """Get (H, W) from shape_meta for the first RGB obs. shape_meta['obs'][key]['shape'] is [C, H, W]."""
    obs_meta = shape_meta.get("obs", {})
    if render_obs_key not in obs_meta:
        return (84, 84)
    shape = obs_meta[render_obs_key].get("shape")
    if not shape or len(shape) != 3:
        return (84, 84)
    # [C, H, W] -> (H, W)
    return (int(shape[1]), int(shape[2]))


def create_env(
    make_env_fn,
    use_pointcloud_obs=False,
    shape_meta=None,
    render_obs_key="head_image",
    init_perturb=None,
):
    kwargs = dict(use_pointcloud_obs=use_pointcloud_obs)
    if init_perturb is not None:
        kwargs["init_perturb"] = bool(init_perturb)
    if shape_meta is not None:
        kwargs["resolution"] = _resolution_from_shape_meta(shape_meta, render_obs_key)
    return _resolve_callable(make_env_fn)(**kwargs)


class BigymImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            make_env_fn,
            shape_meta: dict,
            n_train=8,
            n_train_vis=3,
            n_test=16,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='head_image',
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            use_pointcloud_obs=False,
            init_perturb=None,
            disable_tqdm=False,
            use_ddp_async_env=False,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        def env_fn():
            raw_env = create_env(
                make_env_fn,
                use_pointcloud_obs=use_pointcloud_obs,
                shape_meta=shape_meta,
                render_obs_key=render_obs_key,
                init_perturb=init_perturb,
            )
            control_frequency = getattr(raw_env, "control_frequency", None)
            if control_frequency is None:
                raise ValueError("BiGym env must expose control_frequency")
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    BigymImageWrapper(
                        env=raw_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=control_frequency,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=1
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        # dummy env_fn for initializing spaces without rendering context
        def dummy_env_fn():
            raw_env = create_env(
                make_env_fn,
                use_pointcloud_obs=use_pointcloud_obs,
                shape_meta=shape_meta,
                render_obs_key=render_obs_key,
                init_perturb=init_perturb,
            )
            control_frequency = getattr(raw_env, "control_frequency", None)
            if control_frequency is None:
                raise ValueError("BiGym env must expose control_frequency")
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    BigymImageWrapper(
                        env=raw_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=control_frequency,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=1
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train initializations
        for i in range(n_train):
            enable_render = i < n_train_vis

            def init_fn(env, seed=i, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath('media', wandb.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.seed(seed)

            env_seeds.append(i)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test initializations
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath('media', wandb.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        async_env_cls = AsyncVectorEnvDDP if use_ddp_async_env else AsyncVectorEnv
        env = async_env_cls(env_fns, dummy_env_fn=dummy_env_fn)
        control_frequency = env.call("control_frequency")[0]

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = control_frequency
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.max_rewards = {}
        self.shape_meta = shape_meta
        self.control_frequency = control_frequency
        self.use_pointcloud_obs = use_pointcloud_obs
        self.init_perturb = bool(init_perturb)
        self.disable_tqdm = bool(disable_tqdm)
        self.use_ddp_async_env = bool(use_ddp_async_env)
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0

    def run(self, policy: BaseImagePolicy):
        # Infer device from actual parameters to avoid stale policy.device in subprocesses.
        device = next(policy.parameters()).device
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

            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc="Eval BiGymImage {} / {}".format(chunk_idx + 1, n_chunks), leave=False, mininterval=self.tqdm_interval_sec, disable=self.disable_tqdm)

            done = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.as_tensor(x, device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix + 'max_score'] = self.max_rewards[prefix]

        return log_data
