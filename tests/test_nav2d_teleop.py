"""Keyboard commands must move the robot in the advertised direction."""

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.nav2d.nav_env_cfg import turtlebot4_nav_env_cfg
from mjlab.tasks.nav2d.teleop import (
  KeyboardDriveViewer,
  KeyboardVelocityPolicy,
  velocity_to_action,
)
from mjlab.viewer.native.keys import (
  KEY_DOWN,
  KEY_LEFT,
  KEY_RIGHT,
  KEY_SPACE,
  KEY_TAB,
  KEY_UP,
)


def test_keys_limits_and_stop():
  policy = KeyboardVelocityPolicy("cpu")
  policy.handle_key(KEY_UP)
  policy.handle_key(KEY_LEFT)
  assert policy.command == (0.05, 0.2)
  policy.handle_key(KEY_DOWN)
  policy.handle_key(KEY_RIGHT)
  assert policy.command == (0.0, 0.0)
  for _ in range(100):
    policy.handle_key(KEY_UP)
    policy.handle_key(KEY_RIGHT)
  assert policy.command == (0.3, -1.5)
  assert policy(None).abs().max() <= 1
  policy.handle_key(KEY_SPACE)
  assert policy.command == (0.0, 0.0)
  policy.handle_key(KEY_DOWN)
  policy.reset()
  assert policy.command == (0.0, 0.0)
  assert not policy.handle_key(-1)


def test_wheel_limit_preserves_curvature():
  v, w = 0.3, 1.5
  left, right = velocity_to_action(v, w)
  assert max(abs(left), abs(right)) == 1.0
  actual_v = -(left + right) * 10 * 0.036 / 2
  actual_w = (left - right) * 10 * 0.036 / 0.23
  assert actual_w / actual_v == pytest.approx(w / v)


def test_aligned_start_forward_and_left_turn_gpu():
  if not torch.cuda.is_available():
    pytest.skip("CUDA required")
  cfg = turtlebot4_nav_env_cfg(play=True)
  cfg.terminations = {}
  with torch.inference_mode():
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
      env.reset()
      robot = env.scene["robot"]
      assert abs(robot.data.heading_w.item()) < 1e-6
      action = torch.tensor([velocity_to_action(0.18, 0.0)], device=env.device)
      start = robot.data.root_link_pos_w.clone()
      for _ in range(30):
        env.step(action)
      assert (robot.data.root_link_pos_w - start)[0, 0] > 0.03
      env.reset()
      assert abs(robot.data.heading_w.item()) < 1e-6
      action = torch.tensor([velocity_to_action(0.0, 0.6)], device=env.device)
      for _ in range(30):
        env.step(action)
      assert robot.data.heading_w.item() > 0.1
    finally:
      env.close()


def test_drive_keys_do_not_trigger_viewer_step_or_pause(monkeypatch):
  viewer = object.__new__(KeyboardDriveViewer)
  viewer.drive_policy = KeyboardVelocityPolicy("cpu")
  pauses = []
  monkeypatch.setattr(viewer, "request_toggle_pause", lambda: pauses.append(True))
  viewer._safe_key_callback(KEY_UP)
  viewer._safe_key_callback(KEY_RIGHT)
  assert viewer.drive_policy.command == (0.05, -0.2)
  viewer._safe_key_callback(KEY_SPACE)
  assert not pauses
  assert viewer.drive_policy.command == (0.0, 0.0)
  viewer._safe_key_callback(KEY_TAB)
  assert pauses == [True]
