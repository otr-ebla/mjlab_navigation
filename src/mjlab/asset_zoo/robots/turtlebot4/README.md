# TurtleBot4 Standard in MuJoCo

This model uses the official TurtleBot4 Standard / OAK-D Pro visual meshes,
converted from Collada to STL. Large meshes are reduced to 12,000 triangles per
material. The source mesh colors and URDF mounting transforms are retained.
Simple collision shapes and the existing two-wheel dynamics are kept for RL;
this is not a calibrated hardware dynamics model.

## Navigation: only LiDAR enabled

`tasks/nav2d/nav_env_cfg.py` selects:

```python
sensors = get_turtlebot4_sensor_cfgs(Turtlebot4SensorsCfg(lidar=True))
```

All other flags default to `False`. Camera frames and sensor mounting sites are
always present in the robot model, but disabled sensors allocate no camera image
buffers and add no builtin IMU/encoder measurements. Navigation retains its
72-beam, 6 m LiDAR approximation. It is attached at the official LiDAR position.
The raycast uses collision geometry, excluding visual meshes and the parent body.
Goal, base velocity, and previous-action policy inputs are simulator state, not
additional enabled hardware sensors.

Preview without a checkpoint:

```bash
uv run play Mjlab-Nav-Flat-Turtlebot4 --agent zero --num-envs 1 --viewer native
```

## Keyboard driving

```bash
uv run python -m mjlab.tasks.nav2d.teleop
```

Click the native viewer to give it keyboard focus. Up/down change the forward
velocity target by 0.05 m/s; left/right change the counterclockwise angular
velocity target by 0.2 rad/s. Targets are bounded to ±0.3 m/s and ±1.5 rad/s.
Commands persist after key release: **Space stops both velocities**. Enter resets
and stops; Tab stops and toggles pause. Current targets and wheel-limited targets
are displayed in the viewer. Wheel saturation scales both wheel speeds together
to retain the requested turning curvature. Actual motion depends on the physics.

Playback and keyboard driving start aligned with world +X. Training keeps random
initial yaw. Keyboard driving uses one environment, LiDAR only, and disables
automatic episode resets so crossing a goal or a timeout does not teleport the
robot. Close the window or press Ctrl+C to quit.

The existing scene is still an open plane. This robot change does not add a room.

## Optional sensor suite

Import `Turtlebot4SensorsCfg` and `get_turtlebot4_sensor_cfgs` from
`mjlab.asset_zoo.robots.turtlebot4.sensors`. Enable the flags you need when
constructing `SceneCfg.sensors`. Start with one environment when enabling cameras.

| Flag | Scene sensor name(s) | Output / approximation |
| --- | --- | --- |
| `lidar` | `lidar` | Horizontal range scan; misses are -1 in raw data |
| `rgbd` | `oakd_rgbd` | RGB and ideal rendered depth |
| `stereo` | `oakd_left`, `oakd_right` | Two RGB images at the official 75 mm baseline; convert to grayscale if needed |
| `imu` | `robot/base_accelerometer`, `robot/base_gyro`, `robot/oakd_accelerometer`, `robot/oakd_gyro` | Ideal acceleration and angular velocity at both IMU frames |
| `wheel_encoders` | `robot/left_encoder`, `robot/right_encoder`, plus `_velocity` | Wheel angle and angular velocity |
| `bumper` | `bumper` | Chassis contact count, force, and position; not five discrete hardware bumper switches |
| `cliff` | `cliff` | Four short range rays at the official cliff sensor frames |
| `ir_proximity` | `ir_proximity` | Seven front proximity ranges; not calibrated IR intensity |
| `optical_mouse` | `robot/optical_mouse_velocity` | Ideal local velocity at the optical mouse frame; no texture-dependent optical flow |
| `wheel_contact` | `wheel_contact` | Contact count / air time for both wheels; a wheel-drop proxy, not suspension switch simulation |

Read outputs through `env.scene[name].data`. For cameras use `.data.rgb` and
`.data.depth`; for ray sensors use `.data.distances`. Enabling a sensor does not
automatically insert its output into policy observations.

The suite supplies simulation equivalents for navigation sensing, not the full
Create 3 firmware. Dock IR opcode reception, battery/charging telemetry, button
states, sensor noise/latency, stereo reconstruction and wheel-drop suspension are
not emulated. The official docking receiver and button frames are retained as
mount sites for future task-specific implementations. RGB camera field of view
matches the upstream simulation approximation, not a calibrated camera.

## Asset provenance and reproduction

- TurtleBot4: https://github.com/turtlebot/turtlebot4
  revision `a6ee13b63cbd524500cc6a68cd20dbef69f326a9` (humble).
  TurtleBot4 description by Clearpath Robotics; Apache-2.0, see
  `licenses/turtlebot4-LICENSE`.
- Create 3: https://github.com/iRobotEducation/create3_sim
  revision `6a8bc44f9be2c260bc17e753c31455d6759af238` (humble).
  Copyright 2021 iRobot Corporation; BSD-3-Clause, see
  `licenses/create3-LICENSE`.

Changes: mesh format conversion, triangle reduction, material color extraction,
fixed transforms flattened into one MuJoCo body, and a 5 cm reference-frame
translation to preserve this project's existing chassis frame. The collision
shapes, actuators, and sensor implementations are local simulation adaptations.

From the mjlab repository, with the source repositories checked out at the above
revisions:

```bash
uv run --with xacro --with pycollada --with fast-simplification \
  python scripts/assets/convert_turtlebot4.py /path/to/turtlebot4 /path/to/create3_sim
```

The generated `visual_assets.xml`, `visual_geoms.xml`, and `sensor_mounts.xml`
are included by `xmls/turtlebot4.xml`. Runtime requires neither ROS nor xacro.
