from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .terrain_loader import TerrainLoader
except ImportError:
    from utils.terrain_loader import TerrainLoader


class DummyGym:
    def __init__(self):
        self.calls = []

    def add_triangle_mesh(self, sim, vertices, triangles, params):
        self.calls.append((sim, vertices, triangles, params))


def _write_obj(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )


def _build_yaml(tmp_path: Path) -> Path:
    mesh_a = tmp_path / "mesh_a.obj"
    mesh_b = tmp_path / "mesh_b.obj"
    _write_obj(mesh_a)
    _write_obj(mesh_b)

    yaml_path = tmp_path / "terrains.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "- name: t0",
                f"  file_path: {mesh_a}",
                "  orientation: x",
                "  size: [2.0, 3.0]",
                "  starts: [[0, 0, 0], [1, 0, 0], [2, 0, 0]]",
                "  goals: [[0, 1, 0], [1, 1, 0], [2, 1, 0]]",
                "  difficulty: 0",
                "- name: t1",
                f"  file_path: {mesh_b}",
                "  orientation: -y",
                "  size: [1.0, 2.0]",
                "  starts: [[0, 0, 1], [1, 0, 1], [2, 0, 1]]",
                "  goals: [[0, 1, 1], [1, 1, 1], [2, 1, 1]]",
                "  difficulty: 1",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def _build_variable_yaml(tmp_path: Path) -> Path:
    mesh_a = tmp_path / "mesh_var_a.obj"
    mesh_b = tmp_path / "mesh_var_b.obj"
    _write_obj(mesh_a)
    _write_obj(mesh_b)

    yaml_path = tmp_path / "terrains_variable.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "- name: t0",
                f"  file_path: {mesh_a}",
                "  orientation: x",
                "  size: [2.0, 3.0]",
                "  starts: [[0, 0, 0], [1, 0, 0]]",
                "  goals: [[0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0]]",
                "  middle_points: [[0.5, 0.5, 0], [1.5, 0.5, 0]]",
                "  difficulty: 0",
                "- name: t1",
                f"  file_path: {mesh_b}",
                "  orientation: -y",
                "  size: [1.0, 2.0]",
                "  starts: [[0, 0, 1], [1, 0, 1], [2, 0, 1], [0.5, 0.5, 1]]",
                "  goals: [[0, 1, 1]]",
                "  difficulty: 1",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def test_loader_builds_expected_tensors_and_offsets(tmp_path):
    yaml_path = _build_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=2,
        num_cols=2,
        gap=1.0,
        terrains_yaml=str(yaml_path),
        device="cpu",
        static_friction=0.8,
        dynamic_friction=0.7,
        restitution=0.1,
    )

    loader = TerrainLoader(cfg=cfg)

    assert len(loader.terrain_instances) == 4
    assert loader.get_terrain_starts().shape == (4, 3, 3)
    assert loader.get_terrain_goals().shape == (4, 3, 3)
    assert loader.get_terrain_difficulties().shape == (4,)

    # step_x = max(size_x)+gap = 2+1 = 3, step_y = max(size_y)+gap = 3+1 = 4
    starts_00 = loader.terrain_instances[0].starts
    offset_00 = loader.terrain_instances[0].offset
    offset_01 = loader.terrain_instances[1].offset
    offset_10 = loader.terrain_instances[2].offset
    array_offset_01 = loader.terrain_instances[1].array_offset
    array_offset_10 = loader.terrain_instances[2].array_offset
    local_offset_00 = loader.terrain_instances[0].local_offset
    local_offset_01 = loader.terrain_instances[1].local_offset
    local_offset_10 = loader.terrain_instances[2].local_offset
    assert torch.allclose(starts_00, loader.get_terrain_starts()[0])
    assert torch.allclose(starts_00[:, :2], torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    assert torch.allclose(offset_00, local_offset_00)
    assert torch.allclose(array_offset_01, torch.tensor([0.0, 4.0, 0.0]))
    assert torch.allclose(array_offset_10, torch.tensor([3.0, 0.0, 0.0]))
    assert torch.allclose(offset_01, array_offset_01 + local_offset_01)
    assert torch.allclose(offset_10, array_offset_10 + local_offset_10)


def test_sequential_sampling_wraparound(tmp_path):
    yaml_path = _build_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=1,
        num_cols=1,
        gap=0.5,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)
    terrain_ids = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    idx = loader.sample_start_indices(terrain_ids, sequential=True)
    assert idx.shape == (4, 1, 1)
    assert torch.allclose(idx.view(-1), torch.tensor([0.0, 1.0, 2.0, 0.0]))

    sampled = loader.sample_starts(terrain_ids, indices=idx)
    assert sampled.shape == (4, 1, 3)
    expected = loader.get_terrain_starts()[0, [0, 1, 2, 0]]
    assert torch.allclose(sampled.squeeze(1), expected)


def test_local_world_coordinate_transforms(tmp_path):
    yaml_path = _build_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=2,
        num_cols=2,
        gap=1.0,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)

    terrain_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    local_points = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    world_points = loader.local_to_world(terrain_ids, local_points)
    assert torch.allclose(
        world_points,
        torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [1.0, 6.0, 1.0],
                [5.0, 1.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(loader.world_to_local(terrain_ids, world_points), local_points)


def test_variable_length_points_are_padded_and_counted(tmp_path):
    yaml_path = _build_variable_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=1,
        num_cols=2,
        gap=0.5,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)

    starts = loader.get_terrain_starts()
    goals = loader.get_terrain_goals()
    num_starts = loader.get_num_starts()
    num_goals = loader.get_num_goals()

    assert starts.shape == (2, 4, 3)
    assert goals.shape == (2, 4, 3)
    assert torch.equal(torch.sort(num_starts).values, torch.tensor([2, 4]))
    assert torch.equal(torch.sort(num_goals).values, torch.tensor([1, 4]))

    for terrain_id, count in enumerate(num_starts.tolist()):
        assert torch.allclose(loader.terrain_instances[terrain_id].starts, starts[terrain_id, :count])
        assert torch.allclose(starts[terrain_id, count:], torch.zeros_like(starts[terrain_id, count:]))

    for terrain_id, count in enumerate(num_goals.tolist()):
        assert torch.allclose(loader.terrain_instances[terrain_id].goals, goals[terrain_id, :count])
        assert torch.allclose(goals[terrain_id, count:], torch.zeros_like(goals[terrain_id, count:]))


def test_optional_middle_points_are_padded_counted_and_sampled(tmp_path):
    yaml_path = _build_variable_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=1,
        num_cols=2,
        gap=0.5,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)

    num_middle_points = loader.get_num_middle_points()
    terrain_with_middle = int(torch.argmax(num_middle_points).item())
    terrain_without_middle = int(torch.argmin(num_middle_points).item())

    assert torch.equal(torch.sort(num_middle_points).values, torch.tensor([0, 2]))
    assert loader.get_terrain_middle_points().shape == (2, 2, 3)
    assert loader.has_middle_points(torch.tensor([terrain_with_middle, terrain_without_middle])).tolist() == [True, False]

    terrain_ids = torch.tensor([terrain_with_middle, terrain_with_middle], dtype=torch.long)
    indices = torch.tensor([[[0.0]], [[1.0]]])
    sampled = loader.sample_middle_points(terrain_ids, indices=indices).squeeze(1)
    expected = loader.get_terrain_middle_points()[terrain_with_middle, [0, 1]]
    assert torch.allclose(sampled, expected)

    empty_sample = loader.sample_middle_points(torch.tensor([terrain_without_middle], dtype=torch.long)).squeeze(1)
    assert torch.allclose(empty_sample, torch.zeros((1, 3)))


def test_variable_length_sequential_sampling_respects_per_terrain_counts(tmp_path):
    yaml_path = _build_variable_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=1,
        num_cols=2,
        gap=0.5,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)
    num_starts = loader.get_num_starts()
    terrain_with_two_starts = int(torch.argmin(num_starts).item())
    terrain_ids = torch.tensor([terrain_with_two_starts] * 5, dtype=torch.long)

    idx = loader.sample_start_indices(terrain_ids, sequential=True)
    assert torch.equal(idx.view(-1).to(dtype=torch.long), torch.tensor([0, 1, 0, 1, 0]))

    sampled = loader.sample_starts(terrain_ids, indices=idx).squeeze(1)
    expected = loader.get_terrain_starts()[terrain_with_two_starts, [0, 1, 0, 1, 0]]
    assert torch.allclose(sampled, expected)


def test_variable_length_index_clamping_uses_per_terrain_counts(tmp_path):
    yaml_path = _build_variable_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=1,
        num_cols=2,
        gap=0.5,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)
    num_goals = loader.get_num_goals()
    terrain_with_one_goal = int(torch.argmin(num_goals).item())
    terrain_ids = torch.tensor([terrain_with_one_goal], dtype=torch.long)
    indices = torch.tensor([[[99.0]]])

    sampled = loader.sample_goals(terrain_ids, indices=indices).squeeze(1)
    expected = loader.get_terrain_goals()[terrain_with_one_goal, [0]]
    assert torch.allclose(sampled, expected)


def test_goal_sequential_sampling_wraparound(tmp_path):
    yaml_path = _build_yaml(tmp_path)
    cfg = SimpleNamespace(
        num_rows=1,
        num_cols=1,
        gap=0.5,
        terrains_yaml=str(yaml_path),
        device="cpu",
    )
    loader = TerrainLoader(cfg=cfg)
    terrain_ids = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    idx = loader.sample_goal_indices(terrain_ids, sequential=True)
    assert idx.shape == (4, 1, 1)
    assert torch.allclose(idx.view(-1), torch.tensor([0.0, 1.0, 2.0, 0.0]))

    sampled = loader.sample_goals(terrain_ids, indices=idx).squeeze(1)
    expected = loader.get_terrain_goals()[0, [0, 1, 2, 0]]
    assert torch.allclose(sampled, expected)


def test_load_into_sim_registers_all_meshes(tmp_path):
    yaml_path = _build_yaml(tmp_path)
    gym = DummyGym()
    cfg = SimpleNamespace(
        num_rows=2,
        num_cols=1,
        gap=0.2,
        terrains_yaml=str(yaml_path),
        device="cpu",
        static_friction=1.2,
        dynamic_friction=1.1,
        restitution=0.05,
    )
    loader = TerrainLoader(cfg=cfg)

    # Avoid dependency on trimesh parsing details in this unit test.
    loader._load_mesh_arrays = lambda *_args, **_kwargs: (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.uint32),
    )
    loader.load_into_sim(sim="dummy_sim", gym=gym)

    assert len(gym.calls) == 2
    for sim, vertices, triangles, params in gym.calls:
        assert sim == "dummy_sim"
        assert len(vertices) == 9
        assert len(triangles) == 3
        assert params.static_friction == 1.2
        assert params.dynamic_friction == 1.1
        assert params.restitution == 0.05
