"""Navigation-level actions for differential-drive robots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DifferentialDriveVelocityActionCfg(ActionTermCfg):
  """Map normalized linear/angular commands to left and right wheel velocity."""

  left_wheel: str = "left_wheel_joint"
  right_wheel: str = "right_wheel_joint"
  wheel_radius: float = 0.036
  wheel_separation: float = 0.23
  max_linear_velocity: float = 0.22
  max_angular_velocity: float = 1.2
  wheel_speed_limit: float = 10.0

  def build(self, env: ManagerBasedRlEnv) -> "DifferentialDriveVelocityAction":
    return DifferentialDriveVelocityAction(self, env)


class DifferentialDriveVelocityAction(ActionTerm):
  """Apply a two-value ``[linear_velocity, angular_velocity]`` command."""

  cfg: DifferentialDriveVelocityActionCfg

  def __init__(self, cfg: DifferentialDriveVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    target_ids, _ = self._entity.find_joints_by_actuator_names(
      (cfg.left_wheel, cfg.right_wheel)
    )
    if len(target_ids) != 2:
      raise ValueError(
        "Differential drive action requires exactly two wheel actuators."
      )
    self._joint_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros((self.num_envs, 2), device=self.device)
    self._wheel_targets = torch.zeros_like(self._raw_actions)

  @property
  def action_dim(self) -> int:
    return 2

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    linear = actions[:, 0].clamp(-1.0, 1.0) * self.cfg.max_linear_velocity
    angular = actions[:, 1].clamp(-1.0, 1.0) * self.cfg.max_angular_velocity
    half_track = self.cfg.wheel_separation / 2
    # The TurtleBot's local wheel axes point along -Y, hence the negative sign.
    left = -(linear - angular * half_track) / self.cfg.wheel_radius
    right = -(linear + angular * half_track) / self.cfg.wheel_radius
    self._wheel_targets[:] = torch.stack((left, right), dim=-1).clamp(
      -self.cfg.wheel_speed_limit, self.cfg.wheel_speed_limit
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_velocity_target(
      self._wheel_targets, joint_ids=self._joint_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._wheel_targets[env_ids] = 0.0
