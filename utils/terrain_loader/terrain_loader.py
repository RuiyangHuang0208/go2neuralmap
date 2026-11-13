from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
import yaml

from tqdm import tqdm


_VALID_ORIENTATIONS = {"x", "-x", "y", "-y"}


@dataclass(frozen=True)
class TerrainSpec:
    name: str
    file_path: Path
    orientation: str
    size: Tuple[float, float]
    starts: torch.Tensor
    goals: torch.Tensor
    middle_points: torch.Tensor
    difficulty: float


@dataclass(frozen=True)
class TerrainInstance:
    terrain_id: int
    row: int
    col: int
    offset: torch.Tensor # World transform (array_offset + local_offset)
    array_offset: torch.Tensor # Grid offset
    local_offset: torch.Tensor # Terrain origin is left-bottom corner
    difficulty: float
    name: str
    file_path: Path
    orientation: str
    size: Tuple[float, float]
    starts: torch.Tensor # Terrain-local bottom-left coordinates
    goals: torch.Tensor # Terrain-local bottom-left coordinates
    middle_points: torch.Tensor # Terrain-local bottom-left coordinates


class TerrainLoader:
    """Load trimesh terrains from YAML and sample starts/goals on a single device."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.dtype = torch.float
        self.num_rows = int(cfg.num_rows)
        self.num_cols = int(cfg.num_cols)
        self.gap = float(cfg.gap)
        self.num_terrains = self.num_rows * self.num_cols

        self._yaml_path = Path(cfg.terrains_yaml).expanduser().resolve()
        self._terrain_specs = self._load_specs(self._yaml_path)
        if len(self._terrain_specs) == 0:
            raise ValueError(f"No terrain entries found in {self._yaml_path}")

        self._selected_specs = self._sample_specs_uniform_difficulty(self._terrain_specs, self.num_terrains)
        self._instances = self._build_instances(self._selected_specs)
        self._build_tensors()
        self.height_resolution = getattr(cfg, "height_resolution", 0.1)
        self._precompute_height_fields()

    @property
    def terrain_instances(self) -> List[TerrainInstance]:
        return self._instances

    def load_into_sim(self, sim: Any, gym: Any) -> None:
        """Load all selected terrain meshes and add them to Isaac Gym."""
        for inst in tqdm(self._instances, desc="Loading Terrains"):
            vertices, triangles = self._load_mesh_arrays(inst.file_path, inst.orientation)
            tm_params = self._build_triangle_mesh_params(vertices.shape[0], triangles.shape[0], inst.offset)
            gym.add_triangle_mesh(
                sim,
                vertices.flatten(order="C"),
                triangles.flatten(order="C"),
                tm_params,
            )

    def get_terrain_difficulties(self) -> torch.Tensor:
        return self._difficulties

    def get_terrain_starts(self) -> torch.Tensor:
        return self._starts

    def get_terrain_goals(self) -> torch.Tensor:
        return self._goals

    def get_terrain_middle_points(self) -> torch.Tensor:
        return self._middle_points

    def get_num_starts(self) -> torch.Tensor:
        return self._num_starts

    def get_num_goals(self) -> torch.Tensor:
        return self._num_goals

    def get_num_middle_points(self) -> torch.Tensor:
        return self._num_middle_points

    def sample_start_indices(self, terrain_ids: torch.Tensor, sequential: bool = False) -> torch.Tensor:
        return self._sample_indices(terrain_ids, sequential=sequential, cursor=self._start_cursor)

    def sample_goal_indices(self, terrain_ids: torch.Tensor, sequential: bool = False) -> torch.Tensor:
        return self._sample_indices(terrain_ids, sequential=sequential, cursor=self._goal_cursor)

    def sample_middle_point_indices(self, terrain_ids: torch.Tensor, sequential: bool = False) -> torch.Tensor:
        return self._sample_indices(terrain_ids, sequential=sequential, cursor=self._middle_point_cursor)

    def sample_starts(
        self,
        terrain_ids: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        sequential: bool = False,
    ) -> torch.Tensor:
        return self._sample_points(self._starts, terrain_ids, indices=indices, sequential=sequential, is_start=True)

    def sample_goals(
        self,
        terrain_ids: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        sequential: bool = False,
    ) -> torch.Tensor:
        return self._sample_points(self._goals, terrain_ids, indices=indices, sequential=sequential, is_start=False)

    def sample_middle_points(
        self,
        terrain_ids: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        sequential: bool = False,
    ) -> torch.Tensor:
        return self._sample_points(
            self._middle_points,
            terrain_ids,
            indices=indices,
            sequential=sequential,
            count_tensor=self._num_middle_points,
            cursor=self._middle_point_cursor,
        )

    def has_middle_points(self, terrain_ids: torch.Tensor) -> torch.Tensor:
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        self._check_terrain_ids(ids)
        return self._num_middle_points.index_select(0, ids) > 0

    def get_terrain_info(self, terrain_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if terrain_ids is None:
            return {
                "difficulties": self._difficulties,
                "starts": self._starts,
                "goals": self._goals,
                "middle_points": self._middle_points,
                "num_starts": self._num_starts,
                "num_goals": self._num_goals,
                "num_middle_points": self._num_middle_points,
            }
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        return {
            "difficulties": self._difficulties[ids],
            "starts": self._starts[ids],
            "goals": self._goals[ids],
            "middle_points": self._middle_points[ids],
            "num_starts": self._num_starts[ids],
            "num_goals": self._num_goals[ids],
            "num_middle_points": self._num_middle_points[ids],
        }

    def get_terrain_origin(self, terrain_ids: torch.Tensor):
        ids = terrain_ids.view(-1)
        origins = []
        N = ids.shape[0]
        for i in range(N):
            id = ids[i]
            origin = self._instances[id].array_offset
            origins.append(origin)

        return torch.stack(origins, dim=0) # (N, 3)

    def local_to_world(self, terrain_ids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        pts = points.to(device=self.device, dtype=self.dtype)
        self._check_terrain_ids(ids)

        if ids.numel() == 0:
            return pts.clone()

        origins = self.get_terrain_origin(ids)
        if pts.ndim == 2:
            return pts + origins
        if pts.ndim == 3:
            return pts + origins.unsqueeze(1)
        raise ValueError(f"Expected points with ndim 2 or 3, got shape {tuple(pts.shape)}")

    def world_to_local(self, terrain_ids: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        pts = points.to(device=self.device, dtype=self.dtype)
        self._check_terrain_ids(ids)

        if ids.numel() == 0:
            return pts.clone()

        origins = self.get_terrain_origin(ids)
        if pts.ndim == 2:
            return pts - origins
        if pts.ndim == 3:
            return pts - origins.unsqueeze(1)
        raise ValueError(f"Expected points with ndim 2 or 3, got shape {tuple(pts.shape)}")

    def get_height(
        self,
        terrain_ids: torch.Tensor,   # (N,)
        local_xy: torch.Tensor,      # (N,2) bottom-left local coords
        clamp: bool = True,
        out_of_bounds: str = "clamp",   # "clamp" or "nan"
    ) -> torch.Tensor:
        """
        Return terrain height z at given local (x,y) for each terrain id.
        - local_xy is per-SubTerrain local coordinate, origin at bottom-left.
        - Uses bilinear interpolation on precomputed heightfield.

        Returns:
          z: (N,) on self.device
        """
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        xy = local_xy.to(device=self.device).view(-1, 2)
        self._check_terrain_ids(ids)

        N = ids.shape[0]
        if N == 0:
            return torch.empty((0,), device=self.device, dtype=self._heightfields.dtype)

        dx = float(self._hf_dx)
        hf = self._heightfields.index_select(0, ids)  # (N, Hmax, Wmax)

        Hmax = hf.shape[1]
        Wmax = hf.shape[2]

        size = self._hf_size.index_select(0, ids)     # (N,2)
        size_x = size[:, 0]
        size_y = size[:, 1]

        x = xy[:, 0].to(dtype=hf.dtype)
        y = xy[:, 1].to(dtype=hf.dtype)

        # ---------- out-of-bounds handling ----------
        if clamp:
            x = torch.clamp(x, torch.zeros_like(size_x), size_x)
            y = torch.clamp(y, torch.zeros_like(size_y), size_y)
            oob_mask = None
        else:
            oob_mask = (x < 0) | (x > size_x) | (y < 0) | (y > size_y)
            if out_of_bounds == "clamp":
                x = torch.clamp(x, 0.0, size_x)
                y = torch.clamp(y, 0.0, size_y)

        # ---------- continuous grid coordinates ----------
        u = x / dx
        v = y / dx

        x0 = torch.floor(u).to(torch.long)
        y0 = torch.floor(v).to(torch.long)

        nx = self._hf_nx.index_select(0, ids)  # (N,)
        ny = self._hf_ny.index_select(0, ids)  # (N,)

        # per-terrain clamp integer indices to valid range [0, nx-1], [0, ny-1]
        zero = torch.zeros_like(x0)
        x0 = torch.minimum(torch.maximum(x0, zero), nx - 1)
        y0 = torch.minimum(torch.maximum(y0, zero), ny - 1)
        x1 = torch.minimum(x0 + 1, nx - 1)
        y1 = torch.minimum(y0 + 1, ny - 1)

        sx = (u - x0.to(u.dtype)).clamp_(0.0, 1.0)
        sy = (v - y0.to(v.dtype)).clamp_(0.0, 1.0)

        # ---------- gather 4 neighbors ----------
        # flatten (N, Hmax, Wmax) -> (N, Hmax*Wmax) to index with linear offsets
        hf_flat = hf.view(N, Hmax * Wmax)

        idx00 = y0 * Wmax + x0
        idx10 = y0 * Wmax + x1
        idx01 = y1 * Wmax + x0
        idx11 = y1 * Wmax + x1

        b = torch.arange(N, device=self.device)
        z00 = hf_flat[b, idx00]
        z10 = hf_flat[b, idx10]
        z01 = hf_flat[b, idx01]
        z11 = hf_flat[b, idx11]

        z = (
            z00 * (1 - sx) * (1 - sy)
            + z10 * sx * (1 - sy)
            + z01 * (1 - sx) * sy
            + z11 * sx * sy
        )

        if (not clamp) and (out_of_bounds == "nan"):
            z = torch.where(oob_mask, torch.full_like(z, float("nan")), z)

        return z

    def _load_specs(self, yaml_path: Path) -> List[TerrainSpec]:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError("Terrain YAML must be a list of terrain entries.")

        specs: List[TerrainSpec] = []
        for entry in data:
            spec = self._parse_entry(entry, yaml_path.parent)
            specs.append(spec)
        return specs

    def _parse_entry(self, entry: Dict[str, Any], base_dir: Path) -> TerrainSpec:
        required = {"name", "file_path", "orientation", "size", "starts", "goals", "difficulty"}
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(f"Terrain entry missing keys: {sorted(missing)}")

        orientation = str(entry["orientation"]).strip()
        if orientation not in _VALID_ORIENTATIONS:
            raise ValueError(f"Unsupported orientation '{orientation}', expected one of {_VALID_ORIENTATIONS}")

        size = entry["size"]
        if not isinstance(size, Sequence) or len(size) != 2:
            raise ValueError("Terrain 'size' must be a sequence with 2 values: [size_x, size_y].")
        size_xy = (float(size[0]), float(size[1]))

        starts = self._to_xyz_tensor(entry["starts"], "starts")
        goals = self._to_xyz_tensor(entry["goals"], "goals")
        middle_points = self._to_optional_xyz_tensor(entry.get("middle_points", []), "middle_points")

        file_path = Path(entry["file_path"]).expanduser()
        if not file_path.is_absolute():
            file_path = (base_dir / file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Terrain mesh file not found: {file_path}")

        return TerrainSpec(
            name=str(entry["name"]),
            file_path=file_path,
            orientation=orientation,
            size=size_xy,
            starts=starts.to(device=self.device, dtype=self.dtype),
            goals=goals.to(device=self.device, dtype=self.dtype),
            middle_points=middle_points.to(device=self.device, dtype=self.dtype),
            difficulty=float(entry["difficulty"]),
        )

    def _to_xyz_tensor(self, values: Any, key: str) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] < 1:
            raise ValueError(
                f"Terrain '{key}' must have shape (K, 3) with K >= 1, got {tuple(tensor.shape)}"
            )
        return tensor

    def _to_optional_xyz_tensor(self, values: Any, key: str) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if tensor.numel() == 0:
            return torch.empty((0, 3), dtype=self.dtype, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError(
                f"Terrain '{key}' must have shape (K, 3), got {tuple(tensor.shape)}"
            )
        return tensor

    def _sample_specs_uniform_difficulty(self, specs: List[TerrainSpec], count: int) -> List[TerrainSpec]:
        by_difficulty: Dict[float, List[TerrainSpec]] = {}
        for spec in specs:
            by_difficulty.setdefault(spec.difficulty, []).append(spec)

        diff_keys = sorted(by_difficulty.keys())
        for key in diff_keys:
            np.random.shuffle(by_difficulty[key])

        difficulty_order = diff_keys.copy()
        np.random.shuffle(difficulty_order)
        positions = {k: 0 for k in diff_keys}

        selected: List[TerrainSpec] = []
        for idx in range(count):
            diff = difficulty_order[idx % len(difficulty_order)]
            pool = by_difficulty[diff]
            pos = positions[diff] % len(pool)
            selected.append(pool[pos])
            positions[diff] += 1
            if (idx + 1) % len(difficulty_order) == 0:
                np.random.shuffle(difficulty_order)
        return selected

    def _build_instances(self, specs: List[TerrainSpec]) -> List[TerrainInstance]:
        max_size_x = max(spec.size[0] for spec in specs)
        max_size_y = max(spec.size[1] for spec in specs)
        step_x = max_size_x + self.gap
        step_y = max_size_y + self.gap

        instances: List[TerrainInstance] = []
        for terrain_id, spec in enumerate(specs):
            row = terrain_id // self.num_cols
            col = terrain_id % self.num_cols
            # offset = torch.tensor([row * step_x, col * step_y, 0.0], dtype=self.dtype, device=self.device)
            # offset = torch.tensor([row * step_x + spec.size[0] // 2, col * step_y + spec.size[1] // 2, 0.0], dtype=self.dtype, device=self.device)
            array_offset = torch.tensor([row * step_x, col * step_y, 0.0], dtype=self.dtype, device=self.device)
            local_offset = torch.tensor([spec.size[0] / 2.0, spec.size[1] / 2.0, 0.0], dtype=self.dtype, device=self.device)
            instances.append(
                TerrainInstance(
                    terrain_id=terrain_id,
                    row=row,
                    col=col,
                    offset=array_offset + local_offset,
                    array_offset=array_offset,
                    local_offset=local_offset,
                    difficulty=spec.difficulty,
                    name=spec.name,
                    file_path=spec.file_path,
                    orientation=spec.orientation,
                    size=spec.size,
                    starts=spec.starts,
                    goals=spec.goals,
                    middle_points=spec.middle_points,
                )
            )
        return instances

    def _build_tensors(self) -> None:
        starts = [inst.starts for inst in self._instances]
        goals = [inst.goals for inst in self._instances]
        middle_points = [inst.middle_points for inst in self._instances]
        difficulties = [inst.difficulty for inst in self._instances]
        start_counts = [start.shape[0] for start in starts]
        goal_counts = [goal.shape[0] for goal in goals]
        middle_point_counts = [points.shape[0] for points in middle_points]
        max_starts = max(start_counts)
        max_goals = max(goal_counts)
        max_middle_points = max(1, max(middle_point_counts))

        padded_starts = torch.zeros((self.num_terrains, max_starts, 3), device=self.device, dtype=self.dtype)
        padded_goals = torch.zeros((self.num_terrains, max_goals, 3), device=self.device, dtype=self.dtype)
        padded_middle_points = torch.zeros(
            (self.num_terrains, max_middle_points, 3),
            device=self.device,
            dtype=self.dtype,
        )
        for terrain_id, start in enumerate(starts):
            padded_starts[terrain_id, : start.shape[0]] = start
        for terrain_id, goal in enumerate(goals):
            padded_goals[terrain_id, : goal.shape[0]] = goal
        for terrain_id, points in enumerate(middle_points):
            padded_middle_points[terrain_id, : points.shape[0]] = points

        self._starts = padded_starts
        self._goals = padded_goals
        self._middle_points = padded_middle_points
        self._difficulties = torch.tensor(difficulties, device=self.device, dtype=self.dtype)
        self._num_starts = torch.tensor(start_counts, device=self.device, dtype=torch.long)
        self._num_goals = torch.tensor(goal_counts, device=self.device, dtype=torch.long)
        self._num_middle_points = torch.tensor(middle_point_counts, device=self.device, dtype=torch.long)
        self._start_cursor = torch.zeros(self.num_terrains, device=self.device, dtype=torch.long)
        self._goal_cursor = torch.zeros(self.num_terrains, device=self.device, dtype=torch.long)
        self._middle_point_cursor = torch.zeros(self.num_terrains, device=self.device, dtype=torch.long)

    def _sample_indices(self, terrain_ids: torch.Tensor, sequential: bool, cursor: torch.Tensor) -> torch.Tensor:
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        if ids.numel() == 0:
            return torch.empty((0, 1, 1), device=self.device, dtype=self.dtype)

        self._check_terrain_ids(ids)

        if cursor is self._start_cursor:
            counts = self._num_starts.index_select(0, ids)
            all_counts = self._num_starts
        elif cursor is self._goal_cursor:
            counts = self._num_goals.index_select(0, ids)
            all_counts = self._num_goals
        else:
            counts = self._num_middle_points.index_select(0, ids)
            all_counts = self._num_middle_points

        if sequential:
            idx = torch.empty_like(ids)
            unique_ids = torch.unique(ids)
            for terrain_id in unique_ids:
                positions = torch.where(ids == terrain_id)[0]
                count = int(positions.numel())
                start = int(cursor[terrain_id].item())
                point_count = int(all_counts[terrain_id].item())
                if point_count <= 0:
                    idx[positions] = 0
                    cursor[terrain_id] = 0
                    continue
                seq = (torch.arange(count, device=self.device, dtype=torch.long) + start) % point_count
                idx[positions] = seq
                cursor[terrain_id] = (start + count) % point_count
        else:
            rand = torch.rand(ids.shape[0], device=self.device)
            idx = torch.floor(rand * counts.to(dtype=rand.dtype)).to(dtype=torch.long)

        return idx.to(dtype=self.dtype).view(-1, 1, 1)

    def _sample_points(
        self,
        points: torch.Tensor,
        terrain_ids: torch.Tensor,
        indices: Optional[torch.Tensor],
        sequential: bool,
        is_start: Optional[bool] = None,
        count_tensor: Optional[torch.Tensor] = None,
        cursor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
        if ids.numel() == 0:
            return torch.empty((0, 1, 3), device=self.device, dtype=self.dtype)

        self._check_terrain_ids(ids)

        if count_tensor is None:
            if is_start is None:
                raise ValueError("Either is_start or count_tensor must be provided.")
            count_tensor = self._num_starts if is_start else self._num_goals
        if cursor is None:
            if is_start is None:
                raise ValueError("Either is_start or cursor must be provided.")
            cursor = self._start_cursor if is_start else self._goal_cursor

        if indices is None:
            indices = self._sample_indices(ids, sequential=sequential, cursor=cursor)
        point_ids = indices.to(device=self.device, dtype=torch.long).view(-1)
        counts = count_tensor.index_select(0, ids)
        valid = counts > 0
        max_point_ids = torch.clamp(counts - 1, min=0)
        point_ids = torch.minimum(torch.maximum(point_ids, torch.zeros_like(point_ids)), max_point_ids)

        batch = torch.zeros((ids.numel(), 3), device=self.device, dtype=self.dtype)
        if torch.any(valid):
            batch[valid] = points[ids[valid], point_ids[valid]]
        return batch.unsqueeze(1)

    def _check_terrain_ids(self, ids: torch.Tensor) -> None:
        if torch.any(ids < 0) or torch.any(ids >= self.num_terrains):
            raise ValueError(
                f"Terrain ids must be within [0, {self.num_terrains - 1}], got min={int(ids.min())}, max={int(ids.max())}"
            )

    def _load_mesh_arrays(self, mesh_path: Path, orientation: str) -> Tuple[np.ndarray, np.ndarray]:
        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.faces, dtype=np.uint32)

        rotation = self._orientation_matrix(orientation)
        vertices = vertices @ rotation.T
        return vertices, triangles

    def _orientation_matrix(self, orientation: str) -> np.ndarray:
        if orientation == "x":
            angle = 0.0
        elif orientation == "-x":
            angle = np.pi
        elif orientation == "y":
            angle = np.pi / 2.0
        elif orientation == "-y":
            angle = -np.pi / 2.0
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")

        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _build_triangle_mesh_params(self, n_vertices: int, n_triangles: int, offset: torch.Tensor) -> Any:
        try:
            from isaacgym import gymapi  # pylint: disable=import-outside-toplevel
        except ImportError:
            params = SimpleNamespace()
            params.transform = SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.0, z=0.0))
        else:
            params = gymapi.TriangleMeshParams()

        params.nb_vertices = int(n_vertices)
        params.nb_triangles = int(n_triangles)
        params.transform.p.x = float(offset[0].item())
        params.transform.p.y = float(offset[1].item())
        params.transform.p.z = float(offset[2].item())
        params.static_friction = float(getattr(self.cfg, "static_friction", 1.0))
        params.dynamic_friction = float(getattr(self.cfg, "dynamic_friction", 1.0))
        params.restitution = float(getattr(self.cfg, "restitution", 0.0))
        return params

    def _precompute_height_fields(self):
        """
        Build padded & stacked heightfields for GPU-parallel sampling.

        Outputs:
          self._heightfields: (T, Hmax, Wmax) torch.float32/float16 on self.device
          self._hf_nx:       (T,) int64 on self.device
          self._hf_ny:       (T,) int64 on self.device
          self._hf_size:     (T,2) float on self.device  (size_x, size_y)
          self._hf_dx:       float  (shared resolution)
          self._hf_Hmax:     int
          self._hf_Wmax:     int
        """
        dx = float(self.height_resolution)
        self._hf_dx = dx

        heightmaps_np = []   # list of (ny, nx) float32
        nx_list = []
        ny_list = []
        size_list = []

        # --------- 1) per-terrain bake (CPU), keep numpy arrays ----------
        # for inst in self._instances:
        #     spec = inst.spec
        for spec in tqdm(self._selected_specs, desc="Computing height field"):
            mesh = trimesh.load(spec.file_path, force="mesh")

            # apply orientation (same as mesh loading path)
            if spec.orientation is not None:
                R = self._orientation_matrix(spec.orientation)  # (3,3) np
                mesh.vertices = mesh.vertices @ R.T

            nx = int(np.ceil(float(spec.size[0]) / dx)) + 1
            ny = int(np.ceil(float(spec.size[1]) / dx)) + 1

            xs = np.linspace(0.0, float(spec.size[0]), nx, dtype=np.float32)
            ys = np.linspace(0.0, float(spec.size[1]), ny, dtype=np.float32)

            heightmap = np.zeros((ny, nx), dtype=np.float32)

            # centered shift: bottom-left local -> centered mesh coords
            cx = float(spec.size[0]) / 2.0
            cy = float(spec.size[1]) / 2.0

            # ray start z: a bit above mesh top
            z0 = float(mesh.bounds[1][2]) + 1.0

            # NOTE: still CPU loops; OK because only for initialization/baking
            for iy, y in enumerate(ys):
                yc = float(y) - cy
                for ix, x in enumerate(xs):
                    xc = float(x) - cx

                    origin = np.array([[xc, yc, z0]], dtype=np.float32)
                    direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)

                    locations, _, _ = mesh.ray.intersects_location(origin, direction)
                    if len(locations) > 0:
                        heightmap[iy, ix] = float(locations[:, 2].max())
                    else:
                        heightmap[iy, ix] = 0.0

            heightmaps_np.append(heightmap)
            nx_list.append(nx)
            ny_list.append(ny)
            size_list.append([float(spec.size[0]), float(spec.size[1])])

        T = len(heightmaps_np)
        if T == 0:
            # keep tensors well-defined
            self._heightfields = torch.empty((0, 0, 0), device=self.device, dtype=torch.float32)
            self._hf_nx = torch.empty((0,), device=self.device, dtype=torch.long)
            self._hf_ny = torch.empty((0,), device=self.device, dtype=torch.long)
            self._hf_size = torch.empty((0, 2), device=self.device, dtype=torch.float32)
            self._hf_Hmax = 0
            self._hf_Wmax = 0
            return

        Hmax = int(max(ny_list))
        Wmax = int(max(nx_list))
        self._hf_Hmax = Hmax
        self._hf_Wmax = Wmax

        # --------- 2) pad to (Hmax,Wmax) and stack ----------
        # pad value: 0.0 (outside valid region won't be sampled if you clamp indices with nx/ny)
        padded = np.zeros((T, Hmax, Wmax), dtype=np.float32)
        for t, hm in enumerate(heightmaps_np):
            ny, nx = hm.shape
            padded[t, :ny, :nx] = hm

        # --------- 3) move to torch (device) + meta tensors ----------
        # choose dtype: if you have self.dtype for sim, you can use it; else keep float32
        hf_dtype = getattr(self, "dtype", torch.float32)
        if hf_dtype not in (torch.float16, torch.float32, torch.bfloat16):
            hf_dtype = torch.float32

        self._heightfields = torch.from_numpy(padded).to(device=self.device, dtype=hf_dtype)
        self._hf_nx = torch.tensor(nx_list, device=self.device, dtype=torch.long)
        self._hf_ny = torch.tensor(ny_list, device=self.device, dtype=torch.long)
        self._hf_size = torch.tensor(size_list, device=self.device, dtype=hf_dtype)
