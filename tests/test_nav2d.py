"""Regression checks for navigation registration, sensing, and reset state."""

from types import SimpleNamespace
from typing import Any, cast

import mujoco
import pytest
import torch

from mjlab.asset_zoo.robots.turtlebot4.sensors import (
  Turtlebot4SensorsCfg,
  get_turtlebot4_sensor_cfgs,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.nav2d import nav_mdp
from mjlab.tasks.nav2d.lidar import PlanarLidarPatternCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


def test_navigation_registered():
  assert "Mjlab-Nav-Flat-Turtlebot4" in list_tasks()
  cfg = load_env_cfg("Mjlab-Nav-Flat-Turtlebot4")
  assert set(cfg.observations) == {"actor", "critic"}
  assert "velocity_command" in cfg.actions


def test_partial_goal_reset_preserves_other_envs_and_uses_origins():
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    scene=SimpleNamespace(
      env_origins=torch.tensor([[0.0, 0.0, 0.0], [16.0, 0.0, 0.0], [0.0, 16.0, 0.0]])
    ),
  )
  nav_mdp.reset_goal_position_uniform(env, None, (2.0, 2.0))
  distances = (env.goal_pos - env.scene.env_origins).norm(dim=-1)
  torch.testing.assert_close(distances, torch.full((3,), 2.0))
  old_goals = env.goal_pos.clone()
  nav_mdp.reset_goal_position_uniform(env, torch.tensor([1]), (3.0, 3.0))
  torch.testing.assert_close(env.goal_pos[[0, 2]], old_goals[[0, 2]])
  torch.testing.assert_close(env.prev_dist, torch.tensor([2.0, 3.0, 2.0]))


def test_lidar_misses_are_clear_space():
  sensor = SimpleNamespace(
    cfg=SimpleNamespace(max_distance=6.0),
    data=SimpleNamespace(distances=torch.tensor([[-1.0, 3.0, 9.0], [0.1, -1.0, -1.0]])),
  )
  env = SimpleNamespace(scene={"lidar": sensor})
  torch.testing.assert_close(nav_mdp.lidar_scan(env)[0], torch.tensor([1.0, 0.5, 1.0]))
  assert nav_mdp.collision_condition(env).tolist() == [False, True]


def test_lidar_pattern_is_horizontal_without_duplicate_beams():
  origins, directions = PlanarLidarPatternCfg().generate_rays(None, "cpu")
  assert origins.shape == (36, 3)
  assert torch.count_nonzero(origins) == 0
  assert torch.count_nonzero(directions[:, 2]) == 0
  torch.testing.assert_close(directions.norm(dim=-1), torch.ones(36))
  assert torch.unique(directions, dim=0).shape[0] == 36


def test_navigation_gpu_rollout_and_partial_reset():
  if not torch.cuda.is_available():
    pytest.skip("GPU rollout requires CUDA")
  cfg = load_env_cfg("Mjlab-Nav-Flat-Turtlebot4")
  cfg.scene.num_envs = 4
  cfg.seed = 42
  with torch.inference_mode():
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    navigation_state = cast(Any, env)
    try:
      obs, _ = env.reset()
      assert isinstance(obs["actor"], torch.Tensor)
      assert obs["actor"].shape == (4, 82)
      assert env.scene.sensor_context is not None
      assert not env.scene.sensor_context.has_cameras
      torch.testing.assert_close(
        nav_mdp.goal_position_robot_frame(env).norm(dim=-1), navigation_state.prev_dist
      )
      start = env.scene["robot"].data.root_link_pos_w.clone()
      actions = torch.full((4, 2), 0.5, device=env.device)
      for _ in range(20):
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(reward).all()
        assert all(
          isinstance(value, torch.Tensor) and torch.isfinite(value).all()
          for value in obs.values()
        )
        assert not terminated.any()
        assert not truncated.any()
      displacement = (env.scene["robot"].data.root_link_pos_w - start)[:, :2]
      assert (displacement.norm(dim=-1) > 0.02).all()
      goals = navigation_state.goal_pos.clone()
      env.reset(env_ids=torch.tensor([1], device=env.device))
      torch.testing.assert_close(navigation_state.goal_pos[[0, 2, 3]], goals[[0, 2, 3]])
      torch.testing.assert_close(
        nav_mdp.goal_position_robot_frame(env).norm(dim=-1)[1],
        navigation_state.prev_dist[1],
      )
    finally:
      env.close()


def test_turtlebot4_navigation_enables_only_lidar():
  cfg = load_env_cfg("Mjlab-Nav-Flat-Turtlebot4")
  assert [sensor.name for sensor in cfg.scene.sensors] == ["lidar"]
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  assert model.nmesh >= 9
  assert model.ncam == 3  # Passive camera frames exist without active camera sensors.
  assert model.nsensor == 0  # No builtin IMU/encoder sensors instantiated.
  for i in range(model.ngeom):
    if model.geom(i).name.startswith("robot/visual_"):
      assert model.geom_contype[i] == 0
      assert model.geom_conaffinity[i] == 0


def _add_sensor_test_wall(spec: mujoco.MjSpec):
  spec.worldbody.add_geom(
    name="sensor_test_wall",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.5, 0.0, 0.25),
    size=(0.02, 1.0, 0.25),
    rgba=(0.8, 0.2, 0.1, 1.0),
  )


def test_turtlebot4_optional_sensors_gpu():
  if not torch.cuda.is_available():
    pytest.skip("GPU sensor test requires CUDA")
  cfg = load_env_cfg("Mjlab-Nav-Flat-Turtlebot4")
  cfg.scene.num_envs = 1
  cfg.scene.spec_fn = _add_sensor_test_wall
  cfg.events["reset_robot"].params["pose_range"] = {}
  cfg.scene.sensors = get_turtlebot4_sensor_cfgs(
    Turtlebot4SensorsCfg(
      rgbd=True,
      stereo=True,
      imu=True,
      wheel_encoders=True,
      bumper=True,
      cliff=True,
      ir_proximity=True,
      optical_mouse=True,
      wheel_contact=True,
      camera_width=32,
      camera_height=24,
    )
  )
  with torch.inference_mode():
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
      env.reset()
      env.step(torch.zeros((1, 2), device=env.device))
      for name in ("oakd_rgbd", "oakd_left", "oakd_right"):
        rgb = env.scene[name].data.rgb
        assert rgb is not None and rgb.shape == (1, 24, 32, 3)
      depth = env.scene["oakd_rgbd"].data.depth
      assert depth is not None and torch.isfinite(depth).all()
      assert 0.3 < depth[0, 12, 16, 0] < 0.8
      assert 0.4 < env.scene["lidar"].data.distances[0, 0] < 0.6
      assert 0.2 < env.scene["ir_proximity"].data.distances[0, 0] < 0.4
      for name in (
        "base_gyro",
        "base_accelerometer",
        "oakd_gyro",
        "oakd_accelerometer",
        "optical_mouse_velocity",
        "left_encoder",
        "right_encoder",
      ):
        assert torch.isfinite(env.scene[f"robot/{name}"].data).all()
      cliffs = env.scene["cliff"].data.distances
      assert cliffs.shape == (1, 4)
      assert ((cliffs > 0) & (cliffs < 0.1)).all()
      assert env.scene["ir_proximity"].data.distances.shape == (1, 7)
      assert env.scene["wheel_contact"].data.found.shape == (1, 2)
      assert env.scene["bumper"].data.found.shape == (1, 1)
    finally:
      env.close()
