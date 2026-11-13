import torch

from utils.diffusion_policy.algorithms.guided_joint_ddpm import (
    GuidedJointDDPMAlgorithm,
    GuidedJointDDPMAlgorithmConfig,
    LambdaRiskScheduleConfig,
)
from utils.diffusion_policy.models import DDPMCommandDenoiserConfig
from utils.diffusion_policy.models.encoders import CommandConditionEncoderConfig
from utils.diffusion_policy.risk import LocalCropCollisionChecker, LocalCropCollisionCheckerConfig


def test_local_crop_collision_checker_detects_obstacle():
    checker = LocalCropCollisionChecker(
        LocalCropCollisionCheckerConfig(
            velocity_scale=1.0,
            obstacle_threshold=70.0,
            robot_radius_cells=0,
            interpolation_step_cells=0.5,
        )
    )
    candidates = torch.zeros((1, 1, 2, 4))
    candidates[..., 0] = 1.0
    candidates[..., 3] = 1.0
    cost_crop = torch.zeros((1, 4, 5, 5))
    cost_crop[:, :, 3, 2] = 100.0

    result = checker.check(
        candidates,
        cost_crop,
        initial_heading=torch.zeros(1),
        target_mask=torch.ones((1, 2), dtype=torch.bool),
        least_count=torch.ones(1),
        velocity_command_dt=torch.ones(1),
    )

    assert result.labels.shape == (1, 1)
    assert result.labels.item() == 1.0
    assert result.checked_path_fraction.item() > 0.0


def test_guided_joint_ddpm_train_step_smoke_and_ema_update():
    torch.manual_seed(0)
    cfg = _make_small_cfg()
    algorithm = GuidedJointDDPMAlgorithm(cfg).to(torch.device("cpu"))
    batch = _make_batch(batch_size=2, horizon=cfg.model.horizon, crop_size=cfg.model.encoder.crop_size)
    optimizer = torch.optim.AdamW(algorithm.model.parameters(), lr=1e-3)

    before = next(algorithm.target_model.parameters()).detach().clone()
    metrics = algorithm.train_step(batch, optimizer, global_step=10)
    after = next(algorithm.target_model.parameters()).detach().clone()

    assert "loss_ddpm" in metrics
    assert "loss_risk" in metrics
    assert "risk_positive_ratio" in metrics
    assert not torch.equal(before, after)


def test_guided_joint_ddpm_inference_interfaces_match_ddpm():
    torch.manual_seed(0)
    cfg = _make_small_cfg()
    algorithm = GuidedJointDDPMAlgorithm(cfg).to(torch.device("cpu"))
    batch = _make_batch(batch_size=2, horizon=cfg.model.horizon, crop_size=cfg.model.encoder.crop_size)

    timesteps = torch.zeros(2, dtype=torch.long)
    noisy = torch.randn_like(batch["target"])
    pred_noise = algorithm.model(batch, noisy, timesteps)
    sample = algorithm.sample(batch, cfg.model.horizon)
    inferred = algorithm.infer(batch, cfg.model.horizon)

    assert pred_noise.shape == batch["target"].shape
    assert sample.shape == batch["target"].shape
    assert inferred.shape == batch["target"].shape


def test_guided_joint_ddpm_infer_uses_online_model_not_target_model():
    torch.manual_seed(0)
    cfg = _make_small_cfg()
    algorithm = GuidedJointDDPMAlgorithm(cfg).to(torch.device("cpu"))
    batch = _make_batch(batch_size=2, horizon=cfg.model.horizon, crop_size=cfg.model.encoder.crop_size)

    def fail_if_target_used(batch):
        raise AssertionError("infer should not use target_model")

    algorithm.target_model.encode_condition = fail_if_target_used
    inferred = algorithm.infer(batch, cfg.model.horizon)

    assert inferred.shape == batch["target"].shape


def test_guided_joint_ddpm_samples_one_candidate_per_dataset_item_by_default():
    torch.manual_seed(0)
    cfg = _make_small_cfg()
    cfg.candidate_sampler.dataset_prob = 1.0
    cfg.candidate_sampler.perturbation_prob = 0.0
    cfg.candidate_sampler.ddpm_prob = 0.0
    algorithm = GuidedJointDDPMAlgorithm(cfg).to(torch.device("cpu"))
    batch = _make_batch(batch_size=2, horizon=cfg.model.horizon, crop_size=cfg.model.encoder.crop_size)
    optimizer = torch.optim.AdamW(algorithm.model.parameters(), lr=1e-3)

    metrics = algorithm.train_step(batch, optimizer, global_step=0)

    assert metrics["candidate_count_per_sample"] == 1.0
    assert metrics["candidate_dataset_fraction"] == 1.0
    assert "loss_risk" in metrics


def test_guided_joint_ddpm_samples_ddpm_only_for_selected_subset():
    torch.manual_seed(0)
    cfg = _make_small_cfg()
    algorithm = GuidedJointDDPMAlgorithm(cfg).to(torch.device("cpu"))
    batch = _make_batch(batch_size=4, horizon=cfg.model.horizon, crop_size=cfg.model.encoder.crop_size)
    optimizer = torch.optim.AdamW(algorithm.model.parameters(), lr=1e-3)

    algorithm._sample_candidate_sources = lambda batch_size, device, allow_ddpm: torch.tensor(
        [2, 0, 1, 0],
        device=device,
    )

    def fake_ddpm_candidates(ddpm_batch, horizon, candidate_count):
        assert ddpm_batch["target"].shape[0] == 1
        assert candidate_count == 1
        return ddpm_batch["target"].unsqueeze(1)

    algorithm.sample_target_x0_candidates = fake_ddpm_candidates
    metrics = algorithm.train_step(batch, optimizer, global_step=0)

    assert metrics["candidate_count_per_sample"] == 1.0
    assert metrics["candidate_ddpm_fraction"] == 0.25


def test_guided_joint_ddpm_eval_respects_sample_eval_batches_flag():
    torch.manual_seed(0)
    cfg = _make_small_cfg()
    cfg.sample_eval_batches = False
    cfg.candidate_sampler.dataset_prob = 0.0
    cfg.candidate_sampler.perturbation_prob = 0.0
    cfg.candidate_sampler.ddpm_prob = 1.0
    algorithm = GuidedJointDDPMAlgorithm(cfg).to(torch.device("cpu"))
    batch = _make_batch(batch_size=2, horizon=cfg.model.horizon, crop_size=cfg.model.encoder.crop_size)

    def fail_if_sampled(batch, horizon, candidate_count):
        raise AssertionError("DDPM candidate sampling should be disabled for eval")

    algorithm.sample_target_x0_candidates = fail_if_sampled
    metrics = algorithm.eval_step(batch)

    assert metrics["candidate_ddpm_fraction"] == 0.0
    assert "loss_risk" in metrics


def _make_small_cfg():
    encoder = CommandConditionEncoderConfig(
        crop_size=4,
        latent_dim=1,
        map_channels=8,
        cnn_channels=(4,),
        vector_hidden_dims=(8,),
        output_dim=8,
    )
    model = DDPMCommandDenoiserConfig(
        encoder=encoder,
        horizon=3,
        command_dim=4,
        timestep_dim=8,
        hidden_dims=(16,),
    )
    cfg = GuidedJointDDPMAlgorithmConfig(
        model=model,
        num_diffusion_steps=8,
        sampling_steps=4,
        candidate_steps=(1, 2, 3, 4),
        lambda_risk=LambdaRiskScheduleConfig(
            start_step=0,
            warmup_steps=1,
            max_value=0.1,
        ),
    )
    cfg.risk_classifier.hidden_dims = (16,)
    cfg.collision_checker.velocity_scale = 1.0
    cfg.collision_checker.robot_radius_cells = 0
    cfg.collision_checker.obstacle_threshold = 70.0
    cfg.ema_tau = 0.5
    return cfg


def _make_batch(batch_size: int, horizon: int, crop_size: int):
    return {
        "map_crop": torch.randn(batch_size, 4, crop_size, crop_size),
        "uncertainty_crop": torch.zeros(batch_size, 4, crop_size, crop_size),
        "goal_delta": torch.randn(batch_size, 2),
        "current_latent": torch.randn(batch_size, 4),
        "current_heading": torch.tensor([[0.0, 1.0]]).repeat(batch_size, 1),
        "initial_heading": torch.zeros(batch_size),
        "target": _make_target(batch_size, horizon),
        "target_mask": torch.ones(batch_size, horizon, dtype=torch.bool),
        "cost_map_crop": torch.zeros(batch_size, 4, crop_size, crop_size),
        "least_count": torch.ones(batch_size),
        "velocity_command_dt": torch.full((batch_size,), 0.5),
    }


def _make_target(batch_size: int, horizon: int):
    target = torch.zeros(batch_size, horizon, 4)
    target[..., 0] = 0.5
    target[..., 3] = 1.0
    return target
