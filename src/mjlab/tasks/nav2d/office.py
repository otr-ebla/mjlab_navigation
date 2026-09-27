"""Static indoor-office geometry and valid navigation reset locations."""

from collections.abc import Iterable

import mujoco

NAVIGATION_POINTS = (
  (0.0, 0.0),
  (-4.0, 0.0),
  (4.0, 0.0),
  (-4.0, 3.0),
  (0.0, 3.0),
  (4.0, 3.0),
  (-4.0, -3.0),
  (0.0, -3.0),
  (4.0, -3.0),
)


def _add_box(
  spec: mujoco.MjSpec,
  name: str,
  pos: tuple[float, float, float],
  size: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
) -> None:
  spec.worldbody.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=pos,
    size=tuple(value / 2 for value in size),
    rgba=rgba,
  )


def _wall_segments(
  spec: mujoco.MjSpec, prefix: str, y: float, segments: Iterable[tuple[float, float]]
) -> None:
  for index, (left, right) in enumerate(segments):
    _add_box(
      spec,
      f"office_{prefix}_{index}",
      ((left + right) / 2, y, 1.3),
      (right - left, 0.16, 2.6),
      (0.78, 0.76, 0.70, 1.0),
    )


def add_office_floor(spec: mujoco.MjSpec) -> None:
  """Add a furnished office with collidable rooms, corridors, and true door gaps."""
  _add_box(
    spec,
    "office_floor_finish",
    (0.0, 0.0, -0.055),
    (12.0, 10.0, 0.10),
    (0.23, 0.28, 0.32, 1.0),
  )
  wall_color = (0.78, 0.76, 0.70, 1.0)
  _add_box(spec, "office_outer_north", (0.0, 5.0, 1.3), (12.0, 0.16, 2.6), wall_color)
  _add_box(spec, "office_outer_south", (0.0, -5.0, 1.3), (12.0, 0.16, 2.6), wall_color)
  _add_box(spec, "office_outer_west", (-6.0, 0.0, 1.3), (0.16, 10.0, 2.6), wall_color)
  _add_box(spec, "office_outer_east", (6.0, 0.0, 1.3), (0.16, 10.0, 2.6), wall_color)
  # The central corridor walls leave 1.2 m door openings at x = -4, 0, and 4.
  segments = ((-6.0, -4.6), (-3.4, -0.6), (0.6, 2.6), (3.4, 6.0))
  _wall_segments(spec, "north_corridor", 1.0, segments)
  _wall_segments(spec, "south_corridor", -1.0, segments)
  for x in (-2.0, 2.0):
    _add_box(
      spec,
      f"office_north_divider_{x:+.0f}",
      (x, 3.0, 1.3),
      (0.16, 4.0, 2.6),
      wall_color,
    )
    _add_box(
      spec,
      f"office_south_divider_{x:+.0f}",
      (x, -3.0, 1.3),
      (0.16, 4.0, 2.6),
      wall_color,
    )
  desk_color = (0.38, 0.20, 0.10, 1.0)
  for index, (x, y) in enumerate(
    ((-4.7, 3.9), (-0.7, 3.9), (3.3, 3.9), (-4.7, -3.9), (-0.7, -3.9), (3.3, -3.9))
  ):
    _add_box(spec, f"office_desk_{index}", (x, y, 0.38), (1.2, 0.65, 0.76), desk_color)


# (min_x, min_y, max_x, max_y) clearance rectangles. Desk areas are excluded.
NAVIGATION_ZONE_BOUNDS = (
  (-5.25, -0.65, 5.25, 0.65),  # Central corridor.
  (-5.25, 1.55, -2.75, 3.25),
  (-1.25, 1.55, 1.25, 3.25),
  (2.75, 1.55, 5.25, 3.25),
  (-5.25, -3.25, -2.75, -1.55),
  (-1.25, -3.25, 1.25, -1.55),
  (2.75, -3.25, 5.25, -1.55),
)
