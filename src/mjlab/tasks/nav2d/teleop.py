"""Arrow-key velocity control: uv run python -m mjlab.tasks.nav2d.teleop.

Arrow presses adjust a latched velocity command. Space stops, Enter resets,
and Tab pauses. The native viewer only exposes key presses, not key releases.
"""

from threading import Lock
from typing import cast

import mujoco
import mujoco.viewer
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.nav2d.nav_env_cfg import turtlebot4_nav_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import EnvProtocol
from mjlab.viewer.native.keys import (
  KEY_DOWN,
  KEY_LEFT,
  KEY_RIGHT,
  KEY_SPACE,
  KEY_TAB,
  KEY_UP,
)

# Dimensions and wheel action scale of the local MJCF (not hardware calibration).
WHEEL_RADIUS = 0.036
WHEEL_SEPARATION = 0.23
WHEEL_SPEED_LIMIT = 10.0


def velocity_to_action(linear: float, angular: float) -> tuple[float, float]:
  """Map robot-forward m/s and counterclockwise rad/s to bounded wheel actions.

  Both joint axes point along -Y in this MJCF, so forward rolling requires
  negative joint velocity. Scale both wheels together to retain curvature.
  """
  left = -(linear - angular * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
  right = -(linear + angular * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
  scale = max(WHEEL_SPEED_LIMIT, abs(left), abs(right))
  return left / scale, right / scale


class KeyboardVelocityPolicy:
  """Thread-safe velocity targets shared by the UI and simulation threads."""

  def __init__(self, device: str):
    self.device = device
    self._lock = Lock()
    self._linear = 0.0
    self._angular = 0.0

  @property
  def command(self) -> tuple[float, float]:
    with self._lock:
      return self._linear, self._angular

  def reset(self) -> None:
    with self._lock:
      self._linear = self._angular = 0.0

  def handle_key(self, key: int) -> bool:
    with self._lock:
      if key == KEY_UP:
        self._linear += 0.05
      elif key == KEY_DOWN:
        self._linear -= 0.05
      elif key == KEY_LEFT:
        self._angular += 0.2
      elif key == KEY_RIGHT:
        self._angular -= 0.2
      elif key == KEY_SPACE:
        self._linear = self._angular = 0.0
      else:
        return False
      self._linear = round(max(-0.3, min(0.3, self._linear)), 6)
      self._angular = round(max(-1.5, min(1.5, self._angular)), 6)
      return True

  def __call__(self, obs: object) -> torch.Tensor:
    del obs
    return torch.tensor([velocity_to_action(*self.command)], device=self.device)


class KeyboardDriveViewer(NativeMujocoViewer):
  def __init__(self, env: ManagerBasedRlEnv, policy: KeyboardVelocityPolicy):
    super().__init__(cast(EnvProtocol, env), policy)
    self.drive_policy = policy

  def _safe_key_callback(self, key: int) -> None:
    # Consume arrows before the normal viewer's right-arrow single-step binding.
    if self.drive_policy.handle_key(key):
      return
    if key == KEY_TAB:
      self.drive_policy.reset()
      self.request_toggle_pause()
      return
    super()._safe_key_callback(key)

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    linear, angular = self.drive_policy.command
    left, right = velocity_to_action(linear, angular)
    applied_v = -(left + right) * WHEEL_SPEED_LIMIT * WHEEL_RADIUS / 2
    applied_w = (left - right) * WHEEL_SPEED_LIMIT * WHEEL_RADIUS / WHEEL_SEPARATION
    viewer.set_texts(
      (
        mujoco.mjtFontScale.mjFONTSCALE_150.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        "Keyboard drive\nLinear target\nAngular target\nWheel-limited target\n"
        "Up / Down\nLeft / Right\nSpace\nEnter\nTab\nCommand persists",
        f"{'PAUSED' if self.get_status().paused else 'RUNNING'}\n"
        f"{linear:+.2f} m/s\n{angular:+.2f} rad/s\n"
        f"{applied_v:+.2f} m/s, {applied_w:+.2f} rad/s\n"
        "Change linear speed\nChange turn rate\nStop\nReset pose and stop\n"
        "Pause / resume\nPress Space to stop",
      )
    )


def main() -> None:
  cfg = turtlebot4_nav_env_cfg(play=True)
  cfg.scene.num_envs = 1
  # Manual driving should not teleport on timeout or when crossing a random goal.
  cfg.terminations = {}
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    policy = KeyboardVelocityPolicy(device)
    print("Click the viewer. Arrows adjust v/w; Space stops; Enter resets; Tab pauses.")
    print("Commands persist after releasing a key. Close the window to exit.")
    KeyboardDriveViewer(env, policy).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
