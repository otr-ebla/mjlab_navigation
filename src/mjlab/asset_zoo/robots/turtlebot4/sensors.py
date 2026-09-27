"""Opt-in TurtleBot4 sensors; the default enables only planar LiDAR.

Camera depth and proximity ranges are ideal geometric measurements. They do not
model the OAK-D stereo algorithm or the Create 3 IR intensity/docking firmware.
"""

from dataclasses import dataclass

import mujoco
import torch

from mjlab.sensor import (
  BuiltinSensorCfg,
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  SensorCfg,
)


@dataclass
class PlanarLidarPatternCfg(RingPatternCfg):
  rings: tuple[RingPatternCfg.Ring, ...] = ()
  num_beams: int = 36

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.num_beams < 1:
      raise ValueError("num_beams must be positive")
    angles = torch.arange(self.num_beams, device=device) * (
      2 * torch.pi / self.num_beams
    )
    directions = torch.stack(
      (angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1
    )
    return torch.zeros_like(directions), directions


@dataclass
class Turtlebot4SensorsCfg:
  lidar: bool = True
  rgbd: bool = False
  stereo: bool = False
  imu: bool = False
  wheel_encoders: bool = False
  bumper: bool = False
  cliff: bool = False
  ir_proximity: bool = False
  optical_mouse: bool = False
  wheel_contact: bool = False
  lidar_beams: int = 36
  lidar_range: float = 6.0
  camera_width: int = 160
  camera_height: int = 120


def get_turtlebot4_sensor_cfgs(
  cfg: Turtlebot4SensorsCfg | None = None,
  entity: str = "robot",
) -> tuple[SensorCfg, ...]:
  """Build only enabled sensors; disabled cameras allocate no render buffers.

  Builtin sensor names are prefixed by the entity (e.g. robot/base_gyro).
  Camera, raycast and contact names are unprefixed (e.g. oakd_rgbd, cliff).
  """
  cfg = cfg or Turtlebot4SensorsCfg()
  sensors: list[SensorCfg] = []
  if cfg.lidar:
    sensors.append(
      RayCastSensorCfg(
        name="lidar",
        frame=ObjRef(type="site", name="lidar_site", entity=entity),
        pattern=PlanarLidarPatternCfg(num_beams=cfg.lidar_beams),
        ray_alignment="yaw",
        max_distance=cfg.lidar_range,
        include_geom_groups=(0, 3),
      )
    )
  if cfg.rgbd:
    sensors.append(
      CameraSensorCfg(
        name="oakd_rgbd",
        camera_name=f"{entity}/oakd_rgb",
        width=cfg.camera_width,
        height=cfg.camera_height,
        data_types=("rgb", "depth"),
      )
    )
  if cfg.stereo:
    for side in ("left", "right"):
      sensors.append(
        CameraSensorCfg(
          name=f"oakd_{side}",
          camera_name=f"{entity}/oakd_{side}",
          width=cfg.camera_width,
          height=cfg.camera_height,
          data_types=("rgb",),
        )
      )
  if cfg.imu:
    for name, site in (("base", "imu_link_site"), ("oakd", "oakd_imu_frame_site")):
      obj = ObjRef(type="site", name=site, entity=entity)
      sensors.extend(
        (
          BuiltinSensorCfg(
            name=f"{name}_accelerometer", sensor_type="accelerometer", obj=obj
          ),
          BuiltinSensorCfg(name=f"{name}_gyro", sensor_type="gyro", obj=obj),
        )
      )
  if cfg.wheel_encoders:
    for side in ("left", "right"):
      obj = ObjRef(type="joint", name=f"{side}_wheel_joint", entity=entity)
      sensors.extend(
        (
          BuiltinSensorCfg(name=f"{side}_encoder", sensor_type="jointpos", obj=obj),
          BuiltinSensorCfg(
            name=f"{side}_encoder_velocity", sensor_type="jointvel", obj=obj
          ),
        )
      )
  if cfg.bumper:
    sensors.append(
      ContactSensorCfg(
        name="bumper",
        primary=ContactMatch(mode="geom", pattern="chassis_collision", entity=entity),
        fields=("found", "force", "pos"),
        reduce="netforce",
      )
    )
  if cfg.wheel_contact:
    sensors.append(
      ContactSensorCfg(
        name="wheel_contact",
        primary=ContactMatch(mode="geom", pattern=".*_wheel_collision", entity=entity),
        fields=("found",),
        track_air_time=True,
      )
    )
  if cfg.cliff:
    sensors.append(
      RayCastSensorCfg(
        name="cliff",
        frame=tuple(
          ObjRef(type="site", name=f"cliff_{name}_site", entity=entity)
          for name in ("side_left", "side_right", "front_left", "front_right")
        ),
        pattern=GridPatternCfg(size=(0.0, 0.0), direction=(1.0, 0.0, 0.0)),
        max_distance=0.5,
        include_geom_groups=(0, 3),
      )
    )
  if cfg.ir_proximity:
    sensors.append(
      RayCastSensorCfg(
        name="ir_proximity",
        frame=tuple(
          ObjRef(type="site", name=f"ir_intensity_{name}_site", entity=entity)
          for name in (
            "front_center_left",
            "front_center_right",
            "front_left",
            "front_right",
            "left",
            "right",
            "side_left",
          )
        ),
        pattern=GridPatternCfg(size=(0.0, 0.0), direction=(1.0, 0.0, 0.0)),
        max_distance=0.5,
        include_geom_groups=(0, 3),
      )
    )
  if cfg.optical_mouse:
    sensors.append(
      BuiltinSensorCfg(
        name="optical_mouse_velocity",
        sensor_type="velocimeter",
        obj=ObjRef(type="site", name="mouse_site", entity=entity),
      )
    )
  return tuple(sensors)
