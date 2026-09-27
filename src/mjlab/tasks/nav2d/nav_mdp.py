"""Observations, rewards, and reset events for planar goal navigation."""

from typing import Any

import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .office import NAVIGATION_ZONE_BOUNDS


def _goal_buffers(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
  if not hasattr(env, "goal_pos"):
    env.goal_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env.prev_dist = torch.zeros(env.num_envs, device=env.device)
  return env.goal_pos, env.prev_dist


def goal_position_robot_frame(env: Any) -> torch.Tensor:
  goal, _ = _goal_buffers(env)
  robot = env.scene["robot"]
  diff = goal[:, :2] - robot.data.root_link_pos_w[:, :2]
  yaw = robot.data.heading_w
  c, s = yaw.cos(), yaw.sin()
  return torch.stack(
    (c * diff[:, 0] + s * diff[:, 1], -s * diff[:, 0] + c * diff[:, 1]), dim=-1
  )


def lidar_scan(env: Any) -> torch.Tensor:
  """Normalized 2D LiDAR ranges; misses are max-range free space."""
  sensor = env.scene["lidar"]
  maximum = sensor.cfg.max_distance
  return (
    torch.where(sensor.data.distances < 0, maximum, sensor.data.distances).clamp(
      0, maximum
    )
    / maximum
  )


def progress_to_goal_reward(env: Any) -> torch.Tensor:
  _, previous = _goal_buffers(env)
  distance = goal_position_robot_frame(env).norm(dim=-1)
  progress = previous - distance
  previous.copy_(distance)
  return progress


def goal_reached_condition(env: Any, radius: float = 0.25) -> torch.Tensor:
  return goal_position_robot_frame(env).norm(dim=-1) < radius


def goal_reached_reward(env: Any, radius: float = 0.25) -> torch.Tensor:
  return goal_reached_condition(env, radius).float()


def collision_condition(env: Any, min_dist: float = 0.18) -> torch.Tensor:
  ranges = lidar_scan(env) * env.scene["lidar"].cfg.max_distance
  return ranges.amin(dim=-1) < min_dist


def collision_penalty(env: Any, min_dist: float = 0.18) -> torch.Tensor:
  return collision_condition(env, min_dist).float()


def reset_goal_position_uniform(
  env: Any, env_ids: torch.Tensor | None, radius_range: tuple[float, float] = (2.0, 6.0)
) -> None:
  goal, previous = _goal_buffers(env)
  env_ids = resolve_env_ids(env, env_ids)
  distances = torch.empty(len(env_ids), device=env.device).uniform_(*radius_range)
  angles = torch.empty(len(env_ids), device=env.device).uniform_(-torch.pi, torch.pi)
  goal[env_ids] = env.scene.env_origins[env_ids]
  goal[env_ids, 0] += distances * angles.cos()
  goal[env_ids, 1] += distances * angles.sin()
  previous[env_ids] = distances


def reset_office_navigation_episode(env: Any, env_ids: torch.Tensor | None) -> None:
  """Sample separated start/goal pairs in one open office region.

  Each pair is in the same convex corridor or room clearance region, making the
  goal quasi line-of-sight while requiring the policy to avoid walls and desks.
  """
  env_ids = resolve_env_ids(env, env_ids)
  goal, previous = _goal_buffers(env)
  bounds = torch.tensor(NAVIGATION_ZONE_BOUNDS, device=env.device)
  zone_ids = torch.randint(len(bounds), (len(env_ids),), device=env.device)
  selected = bounds[zone_ids]

  def sample_point() -> torch.Tensor:
    return selected[:, :2] + torch.rand((len(env_ids), 2), device=env.device) * (
      selected[:, 2:] - selected[:, :2]
    )

  start = sample_point()
  target = sample_point()
  for _ in range(8):
    valid = ((target - start).norm(dim=-1) >= 1.0) & (
      (target - start).norm(dim=-1) <= 4.0
    )
    target = torch.where(valid[:, None], target, sample_point())

  robot = env.scene["robot"]
  root_state = robot.data.default_root_state[env_ids].clone()
  root_state[:, :2] = start + env.scene.env_origins[env_ids, :2]
  yaw = torch.empty(len(env_ids), device=env.device).uniform_(-torch.pi, torch.pi)
  root_state[:, 3:7] = quat_from_euler_xyz(
    torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
  )
  robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
  goal[env_ids] = env.scene.env_origins[env_ids]
  goal[env_ids, :2] += target
  previous[env_ids] = (target - start).norm(dim=-1)
