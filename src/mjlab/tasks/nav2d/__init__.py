"""Register TurtleBot 4 navigation."""

from mjlab.tasks.registry import register_mjlab_task

from .nav_env_cfg import (
  turtlebot4_nav_env_cfg,
  turtlebot4_office_nav_env_cfg,
  turtlebot4_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Nav-Flat-Turtlebot4",
  env_cfg=turtlebot4_nav_env_cfg(),
  play_env_cfg=turtlebot4_nav_env_cfg(play=True),
  rl_cfg=turtlebot4_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Nav-Office-Turtlebot4",
  env_cfg=turtlebot4_office_nav_env_cfg(),
  play_env_cfg=turtlebot4_office_nav_env_cfg(play=True),
  rl_cfg=turtlebot4_ppo_runner_cfg(),
)
