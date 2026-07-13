from bigym.action_modes import TorqueActionMode, JointPositionActionMode
from bigym.rby1_cartesian_action_mode_whole_body import RBY1CartesianActionModeWholeBody
from bigym.envs.reach_target import ReachTarget
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.dishwasher_cups import DishwasherLoadCups, DishwasherUnloadCups
from bigym.envs.dishwasher_plates import DishwasherLoadPlates, DishwasherUnloadPlates
from bigym.envs.dishwasher import DishwasherClose
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.robots.configs.rby1 import RBY1
from bigym.envs.manipulation import FlipCup
from bigym.envs.pick_and_place import FlipSandwich, StoreKitchenware

def make_reach_target_env(
    control_frequency=50,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
):
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        init_perturb=init_perturb,
    )
    return env

def make_rby1_reach_target_env(
    control_frequency=50,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return ReachTarget(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),        
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_move_plate_env(
    control_frequency=50,
    low_pass_freq_hz=0.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return MovePlate(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),        
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_move_two_plates_env(
    control_frequency=20,
    low_pass_freq_hz=10.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return MoveTwoPlates(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),        
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_dishwasher_load_cups_env(
    control_frequency=50,
    low_pass_freq_hz=10.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return DishwasherLoadCups(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),        
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_dishwasher_unload_cups_env(
    control_frequency=20,
    low_pass_freq_hz=10.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return DishwasherUnloadCups(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            low_pass_freq_hz=0.0,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_dishwasher_load_plates_env(
    control_frequency=50,
    low_pass_freq_hz=0.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return DishwasherLoadPlates(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_dishwasher_unload_plates_env(
    control_frequency=50,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return DishwasherUnloadPlates(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            low_pass_freq_hz=0.0,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_dishwasher_close_env(
    control_frequency=50,
    low_pass_freq_hz=0.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return DishwasherClose(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )

def make_rby1_flip_cup_env(
    control_frequency=50,
    low_pass_freq_hz=0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=True,
    runtime_hold_steps=10,
):
    return FlipCup(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),        
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )


def make_rby1_flip_sandwich_env(
    control_frequency=20,
    low_pass_freq_hz=10.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return FlipSandwich(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )


def make_rby1_store_kitchenware_env(
    control_frequency=20,
    low_pass_freq_hz=10.0,
    use_pointcloud_obs=False,
    resolution=(84, 84),
    init_perturb=False,
    runtime_hold_steps=0,
):
    return StoreKitchenware(
        action_mode=RBY1CartesianActionModeWholeBody(
            block_until_reached=False,
            direct_mode=False,
            control_frequency=control_frequency,
            interpolation_frequency=control_frequency,
            low_pass_freq_hz=low_pass_freq_hz,
            runtime_hold_steps=runtime_hold_steps,
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig("head", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("left_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
                CameraConfig("right_wrist", rgb=True, depth=False, resolution=resolution, pcd=bool(use_pointcloud_obs), pcd_points=1024, pcd_min_dist=None, pcd_max_dist=3.0),
            ],
        ),
        control_frequency=control_frequency,
        render_mode=None,
        robot_cls=RBY1,
        init_perturb=init_perturb,
    )
