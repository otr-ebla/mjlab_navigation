"""Convert official TurtleBot4 Standard meshes and mounts into MJCF includes.

Run with uv run --with xacro --with pycollada --with fast-simplification
python scripts/assets/convert_turtlebot4.py TB4_REPO CREATE3_REPO.
The source checkouts are read-only. No ROS installation is required.
"""

import argparse
import importlib
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import collada
import numpy as np
import trimesh


def origin(element):
  transform = np.eye(4)
  if element is not None:
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    transform = trimesh.transformations.euler_matrix(
      float(rpy[0]), float(rpy[1]), float(rpy[2])
    )
    transform[:3, 3] = xyz
  return transform


def numbers(values):
  return " ".join(f"{float(value):.9g}" for value in values)


def pose(transform):
  return dict(
    pos=numbers(transform[:3, 3]),
    quat=numbers(trimesh.transformations.quaternion_from_matrix(transform)),
  )


def write_xml(root, path):
  ET.indent(root, space="  ")
  ET.ElementTree(root).write(path, encoding="unicode")


def main():
  xacro = importlib.import_module("xacro")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("turtlebot4", type=Path)
  parser.add_argument("create3", type=Path)
  args = parser.parse_args()
  destination = (
    Path(__file__).resolve().parents[2] / "src/mjlab/asset_zoo/robots/turtlebot4"
  )
  packages = {
    "turtlebot4_description": args.turtlebot4.resolve() / "turtlebot4_description",
    "irobot_create_description": args.create3.resolve()
    / "irobot_create_common/irobot_create_description",
    "irobot_create_control": args.create3.resolve()
    / "irobot_create_common/irobot_create_control",
  }
  with tempfile.TemporaryDirectory() as tmp:
    copies = {name: Path(tmp) / name for name in packages}
    for name, source in packages.items():
      shutil.copytree(source, copies[name])
    for directory in copies.values():
      for path in directory.rglob("*.xacro"):
        text = re.sub(
          r"\$\(find ([^)]+)\)", lambda m: str(copies[m[1]]), path.read_text()
        )
        path.write_text(text)
    robot = ET.fromstring(
      xacro.process_file(
        str(copies["turtlebot4_description"] / "urdf/standard/turtlebot4.urdf.xacro")
      ).toxml()
    )
  transforms = {"base_link": np.eye(4)}
  joints = list(robot.findall("joint"))
  while joints:
    remaining = []
    for joint in joints:
      parent_element = joint.find("parent")
      child_element = joint.find("child")
      if parent_element is None or child_element is None:
        raise ValueError("Joint is missing a parent or child")
      parent, child = parent_element.get("link"), child_element.get("link")
      if parent is None or child is None:
        raise ValueError("Joint is missing a parent or child link name")
      if parent not in transforms:
        remaining.append(joint)
        continue
      transforms[child] = transforms[parent] @ origin(joint.find("origin"))
    if len(remaining) == len(joints):
      raise ValueError("Unresolved URDF tree")
    joints = remaining
  # Existing training model's base frame is 5 cm above the ground.
  shift = np.eye(4)
  shift[2, 3] = -0.05
  assets, geoms, mounts = (
    ET.Element("mujoco"),
    ET.Element("mujoco"),
    ET.Element("mujoco"),
  )
  asset = ET.SubElement(assets, "asset")
  mesh_dir = destination / "meshes"
  mesh_dir.mkdir(exist_ok=True)
  converted = {}
  for link in robot.findall("link"):
    link_name = link.get("name")
    for visual in link.findall("visual"):
      mesh_element = visual.find("geometry/mesh")
      if mesh_element is None:
        continue
      filename = mesh_element.get("filename")
      if filename is None:
        raise ValueError("Visual mesh is missing a filename")
      uri = filename.removeprefix("package://")
      package, relative = uri.split("/", 1)
      path = packages[package] / relative
      if uri not in converted:
        pieces = []
        dae = collada.Collada(str(path))
        if dae.scene is None:
          raise ValueError(f"Mesh has no scene: {path}")
        for geometry in dae.scene.objects("geometry"):
          for primitive in geometry.primitives():
            if hasattr(primitive, "triangleset"):
              primitive = primitive.triangleset()
            mesh = trimesh.Trimesh(
              vertices=primitive.vertex, faces=primitive.vertex_index, process=True
            )
            if len(mesh.faces) > 12000:
              mesh = mesh.simplify_quadric_decimation(face_count=12000)
            name = f"{path.stem}_{len(pieces)}"
            mesh.export(mesh_dir / f"{name}.stl")
            rgba = primitive.material.effect.diffuse
            if not isinstance(rgba, tuple):
              raise ValueError(f"Unsupported textured material in {path}")
            ET.SubElement(asset, "mesh", name=name, file=f"../meshes/{name}.stl")
            pieces.append((name, rgba))
        converted[uri] = pieces
      if link_name is None:
        raise ValueError("Link is missing a name")
      transform = shift @ transforms[link_name] @ origin(visual.find("origin"))
      for index, (mesh_name, rgba) in enumerate(converted[uri]):
        ET.SubElement(
          geoms,
          "geom",
          attrib={
            "name": f"visual_{link_name}_{index}",
            "type": "mesh",
            "mesh": mesh_name,
            "rgba": numbers(rgba),
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "mass": "0",
            **pose(transform),
          },
        )
  for name, transform in transforms.items():
    if name in {"base_link", "left_wheel", "right_wheel"}:
      continue
    ET.SubElement(
      mounts,
      "site",
      attrib={
        "name": f"{name}_site",
        "size": "0.002",
        "group": "4",
        **pose(shift @ transform),
      },
    )
  # LiDAR rays start at the official scan frame, facing robot +X.
  lidar_pose = shift @ transforms["rplidar_link"]
  lidar_pose[:3, :3] = np.eye(3)
  ET.SubElement(
    mounts,
    "site",
    attrib={"name": "lidar_site", "size": "0.002", "group": "4", **pose(lidar_pose)},
  )
  # MuJoCo camera axes: +X image right, +Y image up, -Z viewing direction.
  camera_rotation = np.eye(4)
  camera_rotation[:3, :3] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
  for channel in ("rgb", "left", "right"):
    transform = shift @ transforms[f"oakd_{channel}_camera_frame"] @ camera_rotation
    ET.SubElement(
      mounts,
      "camera",
      attrib={"name": f"oakd_{channel}", "fovy": "46.826", **pose(transform)},
    )
  write_xml(assets, destination / "xmls/visual_assets.xml")
  write_xml(geoms, destination / "xmls/visual_geoms.xml")
  write_xml(mounts, destination / "xmls/sensor_mounts.xml")
  print(
    f"Converted {len(converted)} meshes and {len(transforms)} frames to {destination}"
  )


if __name__ == "__main__":
  main()
