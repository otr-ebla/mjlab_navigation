import math

import torch


class FastLidar2D:
  def __init__(
    self,
    num_envs: int,
    num_beams: int = 36,
    max_range: float = 6.0,
    fov: float = 2 * math.pi,
    device: str = "cuda",
  ):
    self.num_envs = num_envs
    self.num_beams = num_beams
    self.max_range = max_range
    self.device = device

    angles = torch.linspace(-fov / 2, fov / 2, num_beams, device=device)
    self.local_ray_dirs = torch.stack(
      [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
    )

  def compute_ranges(
    self, env_scene, base_pos: torch.Tensor, base_yaw: torch.Tensor
  ) -> torch.Tensor:
    """
    Rotates rays to world orientation and queries batched ray intersections.
    Returns: (num_envs, num_beams) range measurements in [0, max_range].
    """
    # Compute cos/sin of yaw: (num_envs, 1)
    cos_y = torch.cos(base_yaw).unsqueeze(-1)
    sin_y = torch.sin(base_yaw).unsqueeze(-1)

    # Rotate horizontal rays from robot-frame to world-frame
    # shape: (num_envs, num_beams, 3)
    rx = cos_y * self.local_ray_dirs[:, 0] - sin_y * self.local_ray_dirs[:, 1]
    ry = sin_y * self.local_ray_dirs[:, 0] + cos_y * self.local_ray_dirs[:, 1]
    rz = torch.zeros_like(rx)
    ray_dirs = torch.stack([rx, ry, rz], dim=-1)

    # Ray origins at LiDAR height (z = 0.13m)
    ray_origins = base_pos.clone()
    ray_origins[:, 2] += 0.08
    ray_origins = ray_origins.unsqueeze(1).expand(-1, self.num_beams, -1)

    # Query MuJoCo Warp / PhysX scene raycaster
    # For simplicity, fallback distances can be clipped to max_range
    distances = env_scene.raycast(ray_origins, ray_dirs, max_distance=self.max_range)
    return torch.clamp(distances, 0.0, self.max_range)
