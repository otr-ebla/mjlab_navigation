"""Launch the interactive TurtleBot4 office scene."""

import torch

from mjlab.envs import ManagerBasedRlEnv

from .nav_env_cfg import turtlebot4_office_nav_env_cfg
from .teleop import KeyboardDriveViewer, KeyboardVelocityPolicy


def main() -> None:
  cfg = turtlebot4_office_nav_env_cfg(play=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    KeyboardDriveViewer(env, KeyboardVelocityPolicy(device)).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
