"""Convert dexmimicgen HDF5 from delta actions to absolute actions.

Uses robosuite's OSC controller to replay each state and extract the
controller's goal position and orientation.  Supports multiprocessing
for parallel conversion across demos.vscode-webview://1l16i7bji68mbdm8upi5lpm9g0i074g29351ja6l5esde4ubj2e0/mof/scripts/dexmimicgen_dataset_conversion.py

Usage:
    python -m mof.scripts.dexmimicgen_dataset_conversion \
        -i data/dexmimicgen/two_arm_threading.hdf5 \
        -o data/dexmimicgen/two_arm_threading_abs.hdf5 \
        -e data/dexmimicgen/two_arm_threading_eval \
        -n 8
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import copy
import collections
import multiprocessing
import os
import pathlib
import pickle
import shutil

import click
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


class DexMimicGenAbsoluteActionConverter:
    """Convert dexmimicgen delta actions to absolute using env replay.

    Works with robosuite 1.5.x composite controller API.
    Handles dual-arm tasks (two robots).
    """

    def __init__(self, dataset_path, algo_name="bc"):
        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        # Create absolute env meta by switching input_type from delta to absolute
        abs_env_meta = copy.deepcopy(env_meta)
        ctrl_cfg = abs_env_meta["env_kwargs"]["controller_configs"]
        for part_name, part_cfg in ctrl_cfg.get("body_parts", {}).items():
            if "input_type" in part_cfg:
                part_cfg["input_type"] = "absolute"

        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        self.abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        self.n_robots = len(self.env.env.robots)
        assert self.n_robots in (1, 2), "Expected 1 or 2 robots, got %d" % self.n_robots

        self.file = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.file["data"])

    def _find_osc_controller(self, robot):
        """Find the OSC controller part that has goal_pos/goal_ori."""
        for name, ctrl in robot.part_controllers.items():
            if hasattr(ctrl, "goal_pos"):
                return ctrl
        raise RuntimeError("No OSC controller found on robot")

    def convert_actions(self, states, actions):
        """Convert delta action sequence to absolute goal poses in world frame.

        Per-arm layout is auto-detected from action width:
          arm_dim = 7  → parallel-jaw (3 pos + 3 aa + 1 grip)
          arm_dim = 12 → dex hand     (3 pos + 3 aa + 6 grip)

        Only the arm pose is converted (delta → abs via OSC goal_pos/goal_ori);
        gripper channels pass through unchanged because they are joint-space
        targets (parallel grip command or dex hand qpos), not delta poses.
        """
        arm_dim = actions.shape[-1] // self.n_robots
        assert actions.shape[-1] == self.n_robots * arm_dim, (
            "Action width %d not divisible by n_robots=%d"
            % (actions.shape[-1], self.n_robots)
        )
        assert arm_dim in (7, 12), (
            "Unexpected per-arm action width %d; expected 7 (parallel) or 12 (dex)"
            % arm_dim
        )

        stacked = actions.reshape(*actions.shape[:-1], -1, arm_dim)

        goal_pos = np.zeros(stacked.shape[:-1] + (3,), dtype=stacked.dtype)
        goal_ori = np.zeros(stacked.shape[:-1] + (3,), dtype=stacked.dtype)
        gripper = stacked[..., 6:arm_dim]

        env = self.env
        for i in range(len(states)):
            env.reset_to({"states": states[i]})
            for idx, robot in enumerate(env.env.robots):
                robot.control(stacked[i, idx], policy_step=True)
                ctrl = self._find_osc_controller(robot)

                # Controller goal is in robot base frame; transform to world
                base_pos = robot.base_pos  # (3,)
                base_ori = robot.base_ori  # (3, 3)
                goal_pos[i, idx] = base_ori @ ctrl.goal_pos + base_pos
                goal_ori_world = base_ori @ ctrl.goal_ori
                goal_ori[i, idx] = Rotation.from_matrix(goal_ori_world).as_rotvec()

        abs_actions = np.concatenate([goal_pos, goal_ori, gripper], axis=-1)
        return abs_actions.reshape(actions.shape)

    def convert_idx(self, idx):
        demo = self.file["data/demo_%d" % idx]
        states = demo["states"][:]
        actions = demo["actions"][:]
        return self.convert_actions(states, actions)

    def _world_to_robot_base_actions(self, abs_actions_world):
        """Convert world-frame absolute actions to robot-base-frame for eval replay.

        Per-arm width is inferred from the input. Gripper channels (idx >= 6)
        are frame-invariant and pass through unchanged.
        """
        arm_dim = abs_actions_world.shape[-1] // self.n_robots
        stacked = abs_actions_world.reshape(*abs_actions_world.shape[:-1], -1, arm_dim)
        out = stacked.copy()
        env = self.abs_env
        env.reset()
        for idx, robot in enumerate(env.env.robots):
            R_inv = robot.base_ori.T  # (3,3)
            t = robot.base_pos  # (3,)
            # Position: R^{-1} @ (pos_world - t)
            out[..., idx, :3] = np.einsum(
                "ij,...j->...i", R_inv, stacked[..., idx, :3] - t
            )
            # Orientation: R^{-1} @ R_world → axis-angle
            flat_aa = stacked[..., idx, 3:6].reshape(-1, 3)
            R_world = Rotation.from_rotvec(flat_aa).as_matrix()
            R_local = np.einsum("ij,njk->nik", R_inv, R_world)
            out[..., idx, 3:6] = Rotation.from_matrix(R_local).as_rotvec().reshape(
                stacked[..., idx, 3:6].shape
            )
        return out.reshape(abs_actions_world.shape)

    def convert_and_eval_idx(self, idx):
        demo = self.file["data/demo_%d" % idx]
        states = demo["states"][:]
        actions = demo["actions"][:]
        eval_skip_steps = 1

        abs_actions_world = self.convert_actions(states, actions)

        # Convert world-frame abs actions to robot-base-frame for eval replay
        abs_actions_local = self._world_to_robot_base_actions(abs_actions_world)

        # Evaluate both delta and absolute replay
        delta_info = self._evaluate_rollout_error(
            self.env, states, actions, demo, skip=eval_skip_steps
        )
        abs_info = self._evaluate_rollout_error(
            self.abs_env, states, abs_actions_local, demo, skip=eval_skip_steps
        )

        info = {"delta_max_error": delta_info, "abs_max_error": abs_info}
        return abs_actions_world, info

    def _evaluate_rollout_error(self, env, states, actions, demo, skip=1):
        """Replay actions and compare with recorded observations."""
        results = {
            "robot0_eef_pos": [], "robot0_eef_quat": [],
            "robot1_eef_pos": [], "robot1_eef_quat": [],
        }

        for i in range(len(states)):
            env.reset_to({"states": states[i]})
            obs, _, _, _ = env.step(actions[i])
            obs = env.get_observation()
            for key in results:
                if key in obs:
                    results[key].append(obs[key])

        info = {}
        for robot_prefix in ("robot0", "robot1"):
            pos_key = robot_prefix + "_eef_pos"
            quat_key = robot_prefix + "_eef_quat"
            if pos_key not in demo["obs"] or not results[pos_key]:
                continue

            recorded_pos = demo["obs/" + pos_key][:]
            recorded_quat = demo["obs/" + quat_key][:]
            rollout_pos = np.array(results[pos_key])
            rollout_quat = np.array(results[quat_key])

            pos_diff = recorded_pos[1:] - rollout_pos[:-1]
            pos_dist = np.linalg.norm(pos_diff, axis=-1)

            rot_diff = (
                Rotation.from_quat(recorded_quat[1:])
                * Rotation.from_quat(rollout_quat[:-1]).inv()
            )
            rot_dist = rot_diff.magnitude()

            info[robot_prefix + "_pos"] = float(pos_dist[skip:].max())
            info[robot_prefix + "_rot"] = float(rot_dist[skip:].max())

        return info


def _worker(args):
    path, idx, do_eval = args
    converter = DexMimicGenAbsoluteActionConverter(str(path))
    if do_eval:
        abs_actions, info = converter.convert_and_eval_idx(idx)
    else:
        abs_actions = converter.convert_idx(idx)
        info = {}
    return abs_actions, info


@click.command()
@click.option("-i", "--input", required=True, help="Input dexmimicgen HDF5 path")
@click.option("-o", "--output", required=True, help="Output HDF5 path with absolute actions")
@click.option("-e", "--eval_dir", default=None, help="Directory for evaluation metrics")
@click.option("-n", "--num_workers", default=None, type=int, help="Number of parallel workers")
@click.option("--num_demos", default=None, type=int, help="Limit number of demos to convert")
def main(input, output, eval_dir, num_workers, num_demos):
    input = pathlib.Path(input).expanduser()
    assert input.is_file(), "Input file not found: %s" % input
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir(), "Output parent dir must exist: %s" % output.parent

    do_eval = eval_dir is not None
    if do_eval:
        eval_dir = pathlib.Path(eval_dir).expanduser()

    converter = DexMimicGenAbsoluteActionConverter(str(input))
    n_demos = len(converter)
    if num_demos is not None:
        n_demos = min(n_demos, num_demos)
    print("Converting %d demos from %s" % (n_demos, input))

    # Multi-process conversion
    tasks = [(str(input), i, do_eval) for i in range(n_demos)]
    if num_workers == 1:
        results = [_worker(t) for t in tqdm(tasks, desc="Converting")]
    else:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_worker, tasks),
                total=len(tasks),
                desc="Converting",
            ))

    # Copy input and overwrite actions
    print("Copying HDF5 to %s" % output)
    shutil.copy(str(input), str(output))

    with h5py.File(str(output), "r+") as out_file:
        for i in tqdm(range(n_demos), desc="Writing absolute actions"):
            abs_actions, _ = results[i]
            demo = out_file["data/demo_%d" % i]
            demo["actions"][:] = abs_actions

    # Save evaluation
    if do_eval:
        eval_dir.mkdir(parents=True, exist_ok=True)

        infos = [info for _, info in results]
        with open(str(eval_dir / "error_stats.pkl"), "wb") as f:
            pickle.dump(infos, f)

        # Print summary
        metrics = collections.defaultdict(list)
        for info in infos:
            for category, vals in info.items():
                for key, val in vals.items():
                    metrics["%s/%s" % (category, key)].append(val)

        print("\n=== Conversion Error Summary ===")
        for key in sorted(metrics.keys()):
            vals = metrics[key]
            print("  %s: mean=%.6f  max=%.6f" % (key, np.mean(vals), np.max(vals)))

        # Visualization
        try:
            from matplotlib import pyplot as plt
            plt.switch_backend("Agg")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, suffix in zip(axes, ("pos", "rot")):
                for category in ("delta_max_error", "abs_max_error"):
                    key = "%s/robot0_%s" % (category, suffix)
                    if key in metrics:
                        ax.plot(metrics[key], label=category, alpha=0.7)
                ax.legend()
                ax.set_title("robot0_%s" % suffix)
                ax.set_xlabel("Demo index")
            fig.tight_layout()
            fig.savefig(str(eval_dir / "error_stats.png"), dpi=100)
            print("Saved error plot to %s" % (eval_dir / "error_stats.png"))
        except Exception as exc:
            print("Could not generate plot: %s" % exc)

    print("Done!")


if __name__ == "__main__":
    main()
