"""TurtleBot4 navigation task configurations."""

import math

from mjlab.asset_zoo.robots.turtlebot4.sensors import (
  Turtlebot4SensorsCfg,
  get_turtlebot4_sensor_cfgs,
)
from mjlab.asset_zoo.robots.turtlebot4.turtlebot4_constants import get_turtlebot4_cfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import nav_mdp
from .actions import DifferentialDriveVelocityActionCfg
from .config.turtlebot4.rl_cfg import get_turtlebot4_rl_cfg
from .office import add_office_floor


def _navigation_observations() -> dict[str, ObservationTermCfg]:
  return {
    "lidar": ObservationTermCfg(func=nav_mdp.lidar_scan),
    "goal": ObservationTermCfg(func=nav_mdp.goal_position_robot_frame),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "last_command": ObservationTermCfg(func=mdp.last_action),
  }


def turtlebot4_nav_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  terms = _navigation_observations()
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else 6144,
      env_spacing=16.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_turtlebot4_cfg()},
      sensors=get_turtlebot4_sensor_cfgs(
        Turtlebot4SensorsCfg(lidar=True, lidar_beams=72, lidar_range=6.0)
      ),
    ),
    observations={
      "actor": ObservationGroupCfg(terms=terms, enable_corruption=not play),
      "critic": ObservationGroupCfg(terms=dict(terms)),
    },
    actions={
      "velocity_command": DifferentialDriveVelocityActionCfg(entity_name="robot")
    },
    events={
      "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
      "reset_robot": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (0.0, 0.0) if play else (-math.pi, math.pi)}},
      ),
      "reset_goal": EventTermCfg(
        func=nav_mdp.reset_goal_position_uniform,
        mode="reset",
        params={"radius_range": (2.0, 6.0)},
      ),
    },
    rewards={
      "progress": RewardTermCfg(func=nav_mdp.progress_to_goal_reward, weight=10.0),
      "reach_goal": RewardTermCfg(func=nav_mdp.goal_reached_reward, weight=50.0),
      "collision": RewardTermCfg(func=nav_mdp.collision_penalty, weight=-10.0),
      "command_smoothness": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "reach_goal": TerminationTermCfg(func=nav_mdp.goal_reached_condition),
      "collision": TerminationTermCfg(func=nav_mdp.collision_condition),
    },
    sim=SimulationCfg(nconmax=32, njmax=128, mujoco=MujocoCfg(timestep=0.005)),
    decimation=10,
    episode_length_s=1e9 if play else 30.0,
    viewer=ViewerConfig(
      entity_name="robot",
      body_name="base_link",
      distance=1.2 if play else 8.0,
      elevation=-20.0,
      azimuth=45.0,
    ),
  )


def turtlebot4_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return get_turtlebot4_rl_cfg()


def turtlebot4_office_nav_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CUDA-scaled PPO navigation task in the collidable office environment."""
  cfg = turtlebot4_nav_env_cfg(play=play)
  cfg.scene.spec_fn = add_office_floor
  cfg.scene.env_spacing = 14.0
  cfg.events.pop("reset_robot")
  cfg.events.pop("reset_goal")
  cfg.events["reset_office_episode"] = EventTermCfg(
    func=nav_mdp.reset_office_navigation_episode, mode="reset"
  )
  cfg.episode_length_s = 1e9 if play else 35.0
  cfg.viewer.distance = 14.0 if play else 8.0
  cfg.viewer.elevation = -45.0 if play else -20.0
  return cfg


Turtlebot4NavEnvCfg = turtlebot4_nav_env_cfg
