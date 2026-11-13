# General Information
- Module Name: terrain loader
- Class Name: `TerrainLoader`
- Source File: `utils/terrain_loader/terrain_loader.py`
- Description: This module provides loading terrain from trimesh files according to given specifications, and managing curriculum while loading terrains.

# Functionalities

- Load terrains according to a `yaml` file using `trimesh` package.
    - The `yaml` file contains a **list** of terrains.
    - The `yaml` file should contain the following information for each item:
        - `name`: The name of the terrain
        - `file_path`: The path to the trimesh file
        - `orientation`: One of `x`, `-x`, `y`, `-y`
        - `size`: The size of the terrain in the simulation in [m]
        - `starts`: The starting positions for the agents on the terrain in the simulation
        - `goals`: The goal positions for the agents on the terrain in the simulation
        - `difficulty`: The difficulty level of the terrain, which can be used for curriculum learning
    - Current assumption in implementation:
        - each terrain defines one or more starts and goals, each as xyz points with shape `(K, 3)`.

- When initializing the terrain loader (`TerrainLoader(cfg)`)
    - 1 parameter is required:
        - `cfg: LeggedGymCfg.terrain`: The terrain configuration containing curriculum information and terrain YAML path.
            - `cfg.num_rows`: The number of rows of terrains to load
            - `cfg.num_cols`: The number of columns of terrains to load
            - `cfg.gap`: The gap between terrains in [m]
            - `cfg.terrains_yaml`: Path to terrain list YAML file
            - optional: `cfg.device` (default `"cpu"`), `cfg.static_friction`, `cfg.dynamic_friction`, `cfg.restitution`
    - Terrain positions in the simulation are determined by `num_rows`, `num_cols`, and `gap`, with offsets computed from:
        - `step_x = max(size_x among loaded terrains) + gap`
        - `step_y = max(size_y among loaded terrains) + gap`
    - This global offset is added to `starts` and `goals` specified in YAML.
    - Each loaded terrain will be assigned a unique ID, which can be used to manage the terrains and curriculum.
    - If multiple difficulty levels are present in YAML, terrains are selected uniformly across difficulties while filling `num_rows * num_cols` slots (with wraparound when one difficulty has fewer entries).
    - Mesh registration to simulator is done via `load_into_sim(sim, gym)`.

- Get terrain information: The user can get the terrain information
    - Information includes:
        - terrain difficulty
        - terrain starts
        - terrain goals
    - API:
        - `get_terrain_difficulties()`
        - `get_terrain_starts()`
        - `get_terrain_goals()`
        - `get_terrain_info(terrain_ids=None)`

- Sample starts and goals: For specified terrain IDs, the user can sample starts and goals according to YAML.
    - Starts and goals are sampled by **separate methods**:
        - start index sampling: `sample_start_indices(terrain_ids, sequential=False)`
        - goal index sampling: `sample_goal_indices(terrain_ids, sequential=False)`
        - start position sampling: `sample_starts(...)`
        - goal position sampling: `sample_goals(...)`
    - Sampling can be random or sequential.
    - Sequential sampling uses wraparound.
    - Sample index batch shape is `(len_env_ids, 1, 1)`.
    - Returned tensors are on `cfg.device`, dtype `torch.float`.

- Unit tests:
    - `utils/terrain_loader/test_terrain_loader.py`
    - covers initialization, offset construction, wraparound sampling, and mesh registration call behavior.

# References

## Adding Terrains to Gym
```python
from isaacgym import gymutil
from isaacgym import gymapi
import numpy as np
import trimesh

mesh = trimesh.load("mesh_path.obj", force="mesh")
vertices = np.array(mesh.vertices, dtype=np.float32)
triangles = np.array(mesh.faces, dtype=np.uint32)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]

tm_params.transform.p.x = 0.0
tm_params.transform.p.y = 0.0
tm_params.transform.p.z = 0.0
tm_params.static_friction = cfg.static_friction
tm_params.dynamic_friction = cfg.dynamic_friction
tm_params.restitution = cfg.restitution

gym.add_triangle_mesh(sim, vertices.flatten(order='C'), triangles.flatten(order='C'), tm_params)
```

## Usage in GO2RoughEnv

According to `legged_gym/legged_gym/envs/go2/go2_rough_env.py`, `TerrainLoader` is used in the following lifecycle:

1. Create sim and load trimesh terrains in `create_sim()`:
```python
self.cfg.terrain.device = self.device
self.terrain_loader = TerrainLoader(self.cfg.terrain)
self.terrain_loader.load_into_sim(self.sim, self.gym)
```

2. Initialize per-env terrain mapping in `_init_buffers()`:
```python
self.env_terrain_map = torch.zeros((self.num_envs), dtype=torch.long, device=self.device)
self.terrain_difficulties = self.terrain_loader.get_terrain_difficulties()
```

3. Assign terrain IDs to environments (uniform random from candidate terrain ids):
```python
self._assign_terrain_to_env()
```
If curriculum is enabled during reset, candidate terrains are filtered by difficulty:
```python
terrain_ids = torch.where(self.terrain_difficulties < 1)[0]
self._assign_terrain_to_env(env_ids, terrain_ids)
```

4. Sample starts on reset and write to env origins:
```python
sampled = self.terrain_loader.sample_starts(self.env_terrain_map[env_ids]).squeeze(dim=1)
self.env_origins[env_ids, :3] = sampled[:]
```

Integration note:
- `sample_starts(...)` returns shape `(len_env_ids, 1, 3)` in current implementation, so `.squeeze(dim=1)` is used in `GO2RoughEnv` to convert to `(len_env_ids, 3)` before assigning `env_origins`.
