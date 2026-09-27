"""TurtleBot 4 entity with official visuals and simplified collision dynamics."""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

TURTLEBOT4_XML = Path(__file__).parent / "xmls" / "turtlebot4.xml"


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(TURTLEBOT4_XML))


def get_turtlebot4_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*_wheel_joint",)),),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.05),
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )
