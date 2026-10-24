"""Tests for the training control surface: reward shaping, schedules and CLI wiring.

The reward schedule is what makes a from-scratch humanoid baseline learnable: the policy
first learns to stand (posture terms only) and the locomotion terms are annealed in.  These
tests pin the interpolation maths, the runtime plumbing into the vectorised envs and the
CLI parsing that builds a :class:`TrainConfig`.
"""

from __future__ import annotations

import numpy as np
import pytest

from parkour_lab.cli import _build_train_config, build_parser
from parkour_lab.env import ACTION_DIM, EnvConfig, ParkourEnv, RewardConfig
from parkour_lab.levels import LevelCatalog
from parkour_lab.storage import RunStore
from parkour_lab.training.runner import TrainConfig, TrainingRunner, train
from parkour_lab.training.vector import _EnvPool, make_vec_env


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


def test_posture_stage_disables_locomotion_terms() -> None:
    stage = RewardConfig.posture_stage()
    assert stage.progress_per_metre == 0.0
    assert stage.forward_velocity_per_second == 0.0
    assert stage.upright_per_second >= 1.0
    assert stage.height_per_second >= 1.0
    assert stage.success_bonus == RewardConfig().success_bonus


def test_blend_interpolates_every_field() -> None:
    start = RewardConfig.posture_stage()
    end = RewardConfig(progress_per_metre=10.0, forward_velocity_per_second=6.0, fall_penalty=2.0)

    quarter = start.blend(end, 0.25)
    assert quarter.progress_per_metre == pytest.approx(2.5)
    assert quarter.forward_velocity_per_second == pytest.approx(1.5)
    assert quarter.fall_penalty == pytest.approx(0.75 * 10.0 + 0.25 * 2.0)

    assert start.blend(end, 0.0) == start
    assert start.blend(end, 1.0) == end
    # Out-of-range alphas are clamped instead of extrapolating into nonsense weights.
    assert start.blend(end, -3.0) == start
    assert start.blend(end, 7.0) == end


def test_set_reward_updates_cached_envs(catalog: LevelCatalog) -> None:
    base = EnvConfig(level_id="level-01-basics")
    pool = _EnvPool(
        catalog,
        base,
        ["level-01-basics"],
        [0],
    )
    env = pool._current[0]
    assert env.config.reward.progress_per_metre == base.reward.progress_per_metre
    updated = RewardConfig(progress_per_metre=42.0)
    pool.set_reward(updated)
    assert env.config.reward.progress_per_metre == 42.0
    # A level created after the change inherits the new weights too.
    pool.set_levels(["level-02-step-up"])
    pool.step(np.zeros((1, ACTION_DIM)))
    replacement = pool._current[0]
    assert replacement.config.reward.progress_per_metre == 42.0
    pool.close()


def test_set_reward_is_shared_by_the_serial_and_subprocess_envs(catalog: LevelCatalog) -> None:
    levels = ["level-01-basics", "level-02-step-up"]
    reward = RewardConfig(progress_per_metre=3.5, upright_per_second=0.25)
    for workers in (1, 2):
        vec = make_vec_env(
            catalog,
            levels,
            env_config=EnvConfig(level_id=levels[0]),
            seeds=[1, 2],
            workers=workers,
        )
        try:
            vec.set_reward(reward)
            vec.reset()
            # The reward travels through the same code path as any other step, so a
            # consistent pair of rollouts proves every worker picked it up.
            actions = np.zeros((2, ACTION_DIM))
            _, rewards_a, _, _, _ = vec.step(actions)
            _, rewards_b, _, _, _ = vec.step(actions)
            assert np.all(np.isfinite(rewards_a))
            assert np.all(np.isfinite(rewards_b))
        finally:
            vec.close()


def test_runner_anneals_the_reward(catalog: LevelCatalog, tmp_path) -> None:
    start = RewardConfig.posture_stage()
    final = RewardConfig(progress_per_metre=10.0, upright_per_second=0.5)
    config = TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        total_steps=2048,
        rollout_steps=128,
        num_envs=2,
        reward=final,
        reward_start=start,
        reward_ramp_fraction=0.5,
        eval_interval=999,
        checkpoint_interval=999,
        hidden_dim=16,
    )
    store = RunStore.create(tmp_path, run_id="ramp")
    runner = TrainingRunner(config, catalog, store, logger=lambda _msg: None)
    runner.setup()
    try:
        assert runner._active_reward == start
        runner.total_steps = 512  # quarter of the run == half of the ramp
        runner._apply_reward_schedule()
        assert runner._active_reward.progress_per_metre == pytest.approx(5.0)
        assert runner._active_reward.upright_per_second == pytest.approx(
            0.5 * (3.0 + 0.5)
        )
        runner.total_steps = 4096  # past the ramp
        runner._apply_reward_schedule()
        assert runner._active_reward == final
    finally:
        runner.vec.close()


def test_runner_without_a_ramp_keeps_the_final_weights(
    catalog: LevelCatalog, tmp_path
) -> None:
    config = TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        total_steps=512,
        rollout_steps=128,
        num_envs=1,
        eval_interval=999,
        checkpoint_interval=999,
        hidden_dim=16,
    )
    store = RunStore.create(tmp_path, run_id="noramp")
    runner = TrainingRunner(config, catalog, store, logger=lambda _msg: None)
    runner.setup()
    try:
        runner.total_steps = 4096
        runner._apply_reward_schedule()
        assert runner._active_reward == config.reward
    finally:
        runner.vec.close()


def test_config_round_trip_preserves_the_schedule() -> None:
    config = TrainConfig(
        reward=RewardConfig(progress_per_metre=9.0),
        reward_start=RewardConfig.posture_stage(),
        reward_ramp_fraction=0.25,
        max_episode_seconds=6.0,
    )
    restored = TrainConfig.from_dict(config.to_dict())
    assert restored.reward == config.reward
    assert restored.reward_start == config.reward_start
    assert restored.reward_ramp_fraction == pytest.approx(0.25)
    assert restored.max_episode_seconds == pytest.approx(6.0)
    assert restored.env_config("level-01-basics").max_episode_seconds == pytest.approx(6.0)


def test_cli_parses_reward_overrides_and_ramp() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--level",
            "level-01-basics",
            "--steps",
            "5000",
            "--reward-set",
            "progress_per_metre=7.5",
            "--reward-set",
            "forward_velocity_per_second=3",
            "--reward-ramp-fraction",
            "0.2",
            "--reward-start",
            "upright_per_second=4",
        ]
    )
    catalog = LevelCatalog.load()
    config = _build_train_config(args, catalog)
    assert config.reward.progress_per_metre == pytest.approx(7.5)
    assert config.reward.forward_velocity_per_second == pytest.approx(3.0)
    assert config.reward_ramp_fraction == pytest.approx(0.2)
    assert config.reward_start is not None
    assert config.reward_start.upright_per_second == pytest.approx(4.0)
    assert config.reward_start.progress_per_metre == 0.0
    # A ramp fraction of zero means no starting reward, so nothing silently overrides.
    args = parser.parse_args(["train", "--level", "level-01-basics", "--steps", "5000"])
    assert _build_train_config(args, catalog).reward_start is None


def test_cli_rejects_unknown_reward_weights() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["train", "--level", "level-01-basics", "--reward-set", "not_a_weight=1"]
    )
    with pytest.raises(SystemExit, match="unknown reward weight"):
        _build_train_config(args, LevelCatalog.load())

    args = parser.parse_args(
        ["train", "--level", "level-01-basics", "--reward-set", "progress_per_metre"]
    )
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _build_train_config(args, LevelCatalog.load())


def test_episode_cap_shortens_the_time_limit(catalog: LevelCatalog) -> None:
    level = catalog["level-01-basics"]
    capped = ParkourEnv(level, EnvConfig(level_id=level.id, max_episode_seconds=1.0))
    try:
        assert capped.time_limit == pytest.approx(1.0)
        capped.reset(seed=0, options={"deterministic": True})
        done = False
        steps = 0
        while not done:
            result = capped.step(np.zeros(ACTION_DIM))
            steps += 1
            done = result.terminated or result.truncated
        # The cap wins over the level's own 20 s limit: one second of control steps.
        expected = round(1.0 / capped.config.control_dt)
        assert steps == pytest.approx(expected, abs=1)
        assert result.info["termination_reason"] == "timeout"
    finally:
        capped.close()


def test_train_records_the_schedule_in_the_run_config(
    catalog: LevelCatalog, tmp_path
) -> None:
    config = TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        total_steps=512,
        rollout_steps=128,
        num_envs=1,
        eval_interval=999,
        checkpoint_interval=999,
        hidden_dim=16,
        reward_start=RewardConfig.posture_stage(),
        reward_ramp_fraction=0.5,
        max_episode_seconds=4.0,
    )
    summary = train(config, catalog, runs_dir=tmp_path, run_id="schedule-run")
    assert summary["status"] == "completed"
    store = RunStore.load(tmp_path / "schedule-run")
    stored = store.read_config()
    assert stored["reward_ramp_fraction"] == pytest.approx(0.5)
    assert stored["reward_start"]["progress_per_metre"] == 0.0
    assert stored["max_episode_seconds"] == pytest.approx(4.0)


def test_vector_autoreset_returns_the_post_reset_observation(
    catalog: LevelCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trainer must never act on the previous episode's final state.

    ``step`` can only return the state that ended the episode, so the reset observation is
    delivered separately by ``autoreset``.  This test pins that contract down.
    """

    from parkour_lab.env import ParkourEnv
    from parkour_lab.training.vector import SerialVecEnv

    observed_resets: list[np.ndarray] = []
    original_reset = ParkourEnv.reset

    def spy(self, *args, **kwargs):
        observation, info = original_reset(self, *args, **kwargs)
        observed_resets.append(np.asarray(observation, dtype=np.float64).copy())
        return observation, info

    monkeypatch.setattr(ParkourEnv, "reset", spy)
    env = SerialVecEnv(
        catalog,
        ["level-01-basics"],
        env_config=EnvConfig(level_id="level-01-basics", max_episode_seconds=0.3),
    )
    try:
        env.reset()
        infos = None
        for _ in range(200):
            _, _, terminated, truncated, infos = env.step(np.zeros((1, env.action_dim)))
            if terminated[0] or truncated[0]:
                break
        assert infos is not None and (terminated[0] or truncated[0])
        final = np.asarray(infos[0]["final_observation"])
        slots, observations = env.autoreset()
        assert slots.tolist() == [0]
        # autoreset reset the environment; its observation is the one reset returned
        np.testing.assert_allclose(observations[0], observed_resets[-1], atol=1e-6)
        # ... and it is a different state from the one that ended the episode
        assert not np.allclose(observations[0], final, atol=1e-6)
        # nothing left to reset
        slots, observations = env.autoreset()
        assert slots.size == 0 and observations.shape[0] == 0
    finally:
        env.close()


def test_subprocess_vector_env_autoresets_the_same_way(catalog: LevelCatalog) -> None:
    from parkour_lab.training.vector import SerialVecEnv, SubprocVecEnv

    config = EnvConfig(level_id="level-01-basics", max_episode_seconds=0.3)
    serial = SerialVecEnv(catalog, ["level-01-basics"], env_config=config, seeds=[4])
    parallel = SubprocVecEnv(
        catalog, ["level-01-basics"], env_config=config, seeds=[4], workers=1
    )
    try:
        serial.reset()
        parallel.reset()
        for _ in range(200):
            actions = np.zeros((1, serial.action_dim))
            _, _, terminated, truncated, _ = serial.step(actions)
            parallel.step(actions)
            if terminated[0] or truncated[0]:
                break
        serial_slots, serial_obs = serial.autoreset()
        parallel_slots, parallel_obs = parallel.autoreset()
        assert serial_slots.tolist() == parallel_slots.tolist() == [0]
        assert serial_obs.shape == parallel_obs.shape
    finally:
        serial.close()
        parallel.close()
