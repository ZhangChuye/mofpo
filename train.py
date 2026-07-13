"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)  # or absl_logging.FATAL

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from mof.workspace.base_workspace import BaseWorkspace

max_steps = {
    'rby1_reach_target': 400,
    'rby1_move_plate': 400,
    'rby1_move_two_plates': 400,
    'rby1_flip_cup': 400,
    'rby1_flip_sandwich': 800,
    'rby1_store_kitchenware': 800,
    'rby1_dishwasher_load_cups': 800,
    'rby1_dishwasher_load_plates': 1300,
    'rby1_dishwasher_unload_cups': 400,
    'rby1_dishwasher_close': 600,
    # dexmimicgen tasks (from generate_training_config.py)
    'two_arm_threading': 400,
    'two_arm_box_cleanup': 400,
    'two_arm_lift_tray': 750,
    'two_arm_drawer_cleanup': 550,
    'two_arm_three_piece_assembly': 300,
    'two_arm_transport': 1200,
    'two_arm_pouring_humanoid': 400,
    'two_arm_coffee_humanoid': 400,
    'two_arm_can_sort_humanoid': 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.


def get_ws_z_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return 0.9
    else:
        return 0.8

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
OmegaConf.register_new_resolver("get_ws_z_center", get_ws_z_center, replace=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'mof','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
