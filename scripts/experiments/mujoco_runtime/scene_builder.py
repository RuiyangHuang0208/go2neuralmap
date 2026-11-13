from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Optional

import torch
import trimesh
import yaml
from omegaconf import OmegaConf

from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg
from utils.terrain_loader import TerrainLoader


@dataclass
class MuJoCoTerrainScene:
    xml_path: Path
    terrain_loader: TerrainLoader
    work_dir: Path


def build_mujoco_scene_from_terrain_yaml(
    *,
    terrain_cfg_omega: Any,
    robot_xml_path: str | Path,
    work_dir: Optional[str | Path] = None,
    device: torch.device | str = "cpu",
) -> MuJoCoTerrainScene:
    terrain_cfg = _terrain_cfg_object(terrain_cfg_omega, device=device)
    terrain_loader = TerrainLoader(terrain_cfg)

    scene_dir = Path(work_dir).expanduser() if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="go2_mujoco_scene_", dir="/tmp")
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = scene_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    local_robot_xml = _prepare_robot_xml(Path(robot_xml_path).expanduser().resolve(), scene_dir)

    root = ET.Element("mujoco", {"model": "go2 terrain scene"})
    ET.SubElement(root, "include", {"file": str(local_robot_xml)})
    ET.SubElement(root, "statistic", {"center": "0 0 0.1", "extent": "0.8"})
    _add_visual(root)
    asset = ET.SubElement(root, "asset")
    _add_default_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})

    for inst in terrain_loader.terrain_instances:
        mesh_name = f"terrain_{inst.terrain_id}_{_xml_name(inst.name)}"
        mesh_path = mesh_dir / f"{mesh_name}.obj"
        vertices, faces = terrain_loader._load_mesh_arrays(inst.file_path, inst.orientation)
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_path.resolve())})
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": mesh_name,
                "type": "mesh",
                "mesh": mesh_name,
                "pos": _vec3(inst.offset),
                "material": "terrain",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    xml_path = scene_dir / "scene.xml"
    _indent_xml(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return MuJoCoTerrainScene(xml_path=xml_path, terrain_loader=terrain_loader, work_dir=scene_dir)


def build_mujoco_scene_from_config(
    *,
    terrain_cfg_omega: Any,
    mujoco_scene_config_path: str | Path,
    robot_xml_path: str | Path,
    work_dir: Optional[str | Path] = None,
    device: torch.device | str = "cpu",
) -> MuJoCoTerrainScene:
    terrain_cfg = _terrain_cfg_object(terrain_cfg_omega, device=device)
    terrain_loader = TerrainLoader(terrain_cfg)
    config_path = Path(mujoco_scene_config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        scene_cfg = yaml.safe_load(f) or {}
    if not isinstance(scene_cfg, dict):
        raise ValueError("MuJoCo scene config must be a mapping.")

    scene_dir = Path(work_dir).expanduser() if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="go2_mujoco_scene_", dir="/tmp")
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = scene_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    local_robot_xml = _prepare_robot_xml(Path(robot_xml_path).expanduser().resolve(), scene_dir)

    root = ET.Element("mujoco", {"model": str(scene_cfg.get("model", "go2 mujoco terrain scene"))})
    ET.SubElement(root, "include", {"file": str(local_robot_xml)})
    ET.SubElement(root, "statistic", {"center": str(scene_cfg.get("statistic_center", "0 0 0.1")), "extent": str(scene_cfg.get("statistic_extent", "0.8"))})
    _add_visual(root)
    asset = ET.SubElement(root, "asset")
    _add_default_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})

    geoms = scene_cfg.get("geoms", [])
    if not isinstance(geoms, list):
        raise ValueError("MuJoCo scene config 'geoms' must be a list.")
    for geom_index, geom_cfg in enumerate(geoms):
        _add_configured_geom(
            asset=asset,
            worldbody=worldbody,
            geom_cfg=geom_cfg,
            geom_index=geom_index,
            config_dir=config_path.parent,
            mesh_dir=mesh_dir,
        )

    xml_path = scene_dir / "scene.xml"
    _indent_xml(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return MuJoCoTerrainScene(xml_path=xml_path, terrain_loader=terrain_loader, work_dir=scene_dir)


def _terrain_cfg_object(terrain_cfg_omega: Any, *, device: torch.device | str) -> Any:
    defaults = GO2RoughCfg.terrain
    values = {
        "num_rows": int(getattr(defaults, "num_rows", 1)),
        "num_cols": int(getattr(defaults, "num_cols", 1)),
        "gap": float(getattr(defaults, "gap", 1.0)),
        "terrains_yaml": str(getattr(defaults, "terrains_yaml")),
        "static_friction": float(getattr(defaults, "static_friction", 1.0)),
        "dynamic_friction": float(getattr(defaults, "dynamic_friction", 1.0)),
        "restitution": float(getattr(defaults, "restitution", 0.0)),
        "height_resolution": float(getattr(defaults, "height_resolution", 0.1)),
        "device": str(device),
    }
    if terrain_cfg_omega is not None:
        container = OmegaConf.to_container(terrain_cfg_omega, resolve=True) if OmegaConf.is_config(terrain_cfg_omega) else dict(terrain_cfg_omega)
        for key, value in container.items():
            if value is not None:
                values[key] = value
    values["device"] = str(device)
    return SimpleNamespace(**values)


def _add_visual(root: ET.Element) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})


def _add_default_assets(asset: ET.Element) -> None:
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "material", {"name": "terrain", "rgba": "0.45 0.48 0.50 1"})


def _add_configured_geom(
    *,
    asset: ET.Element,
    worldbody: ET.Element,
    geom_cfg: Any,
    geom_index: int,
    config_dir: Path,
    mesh_dir: Path,
) -> None:
    if not isinstance(geom_cfg, dict):
        raise ValueError(f"Geom entry {geom_index} must be a mapping.")
    geom_type = str(geom_cfg.get("type", "box"))
    name = _xml_name(str(geom_cfg.get("name", f"geom_{geom_index}")))
    common = {
        "name": name,
        "type": geom_type,
        "pos": _list_vec(geom_cfg.get("pos", [0.0, 0.0, 0.0])),
        "material": str(geom_cfg.get("material", "terrain")),
        "contype": str(geom_cfg.get("contype", 1)),
        "conaffinity": str(geom_cfg.get("conaffinity", 1)),
    }
    if "quat" in geom_cfg:
        common["quat"] = _list_vec(geom_cfg["quat"])
    if "rgba" in geom_cfg:
        common["rgba"] = _list_vec(geom_cfg["rgba"])
    if "friction" in geom_cfg:
        common["friction"] = _list_vec(geom_cfg["friction"])

    if geom_type in {"plane", "box", "sphere", "capsule", "cylinder", "ellipsoid"}:
        if "size" in geom_cfg:
            common["size"] = _list_vec(geom_cfg["size"])
        ET.SubElement(worldbody, "geom", common)
        return

    if geom_type == "mesh":
        source = geom_cfg.get("file")
        if source is None:
            raise ValueError(f"Mesh geom '{name}' requires 'file'.")
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = (config_dir / source_path).resolve()
        mesh_name = f"{name}_mesh"
        mesh_file = source_path
        if geom_cfg.get("copy", False):
            mesh_file = mesh_dir / source_path.name
            shutil.copy2(source_path, mesh_file)
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_file.resolve())})
        common["mesh"] = mesh_name
        ET.SubElement(worldbody, "geom", common)
        return

    if geom_type == "hfield":
        source = geom_cfg.get("file")
        if source is None:
            raise ValueError(f"Hfield geom '{name}' requires 'file'.")
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = (config_dir / source_path).resolve()
        hfield_name = f"{name}_hfield"
        ET.SubElement(asset, "hfield", {"name": hfield_name, "file": str(source_path), "size": _list_vec(geom_cfg["size"])})
        common.pop("size", None)
        common["hfield"] = hfield_name
        ET.SubElement(worldbody, "geom", common)
        return

    raise ValueError(f"Unsupported MuJoCo geom type '{geom_type}' for geom '{name}'.")


def _vec3(value: torch.Tensor) -> str:
    vals = value.detach().cpu().flatten().tolist()
    return f"{float(vals[0])} {float(vals[1])} {float(vals[2])}"


def _list_vec(values: Any) -> str:
    if isinstance(values, str):
        return values
    return " ".join(str(float(v)) for v in values)


def _xml_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(value))


def _prepare_robot_xml(robot_xml_path: Path, scene_dir: Path) -> Path:
    local_robot_xml = scene_dir / robot_xml_path.name
    shutil.copy2(robot_xml_path, local_robot_xml)
    source_assets = robot_xml_path.parent / "assets"
    target_assets = scene_dir / "assets"
    if target_assets.exists():
        return local_robot_xml
    try:
        target_assets.symlink_to(source_assets, target_is_directory=True)
    except OSError:
        shutil.copytree(source_assets, target_assets)
    return local_robot_xml


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent
