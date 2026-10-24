"""Environment behaviour: rewards, terminations, replay frames and configuration."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from parkour_lab.env import (
    ACTION_DIM,
    OBS_JOINT_POS_START,
    OBSERVATION_DIM,
    OBSERVATION_NAMES,
    EnvConfig,
    ParkourEnv,
    RewardConfig,
    observation_dim,
)
from parkour_lab.levels import LevelCatalog


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


@pytest.fixture()
def env(catalog: LevelCatalog) -> ParkourEnv:
    environment = ParkourEnv(
        catalog["level-01-basics"],
        EnvConfig(level_id="level-01-basics", seed=3, randomize_reset=False),
    )
    yield environment
    environment.close()


def test_observation_layout_is_consistent() -> None:
    assert len(OBSERVATION_NAMES) == 33
    assert OBSERVATION_DIM == 90
    assert observation_dim() == OBSERVATION_DIM
    assert len(set(OBSERVATION_NAMES)) == len(OBSERVATION_NAMES)


def test_reward_components_sum_to_step_reward(env: ParkourEnv) -> None:
    env.reset(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(30):
        result = env.step(rng.uniform(-0.1, 0.1, ACTION_DIM))
        parts = result.info["reward_components"]
        assert set(parts) >= {"progress", "alive", "upright", "action_rate", "torque"}
        if not result.terminated and not result.truncated:
            assert result.reward == pytest.approx(sum(parts.values()), abs=1e-9)
        else:
            assert "terminal" in parts
            assert result.reward == pytest.approx(sum(parts.values()), abs=1e-9)
            break


def test_progress_reward_follows_forward_motion(env: ParkourEnv) -> None:
    env.reset(seed=1)
    env.data.qpos[0] += 0.25
    mujoco.mj_forward(env.model, env.data)
    before = env._previous_goal_distance
    result = env.step(np.zeros(ACTION_DIM))
    assert result.info["reward_components"]["progress"] > 0
    assert env._previous_goal_distance < before


def test_timeout_truncates_at_the_time_limit(catalog: LevelCatalog) -> None:
    level = catalog["level-01-basics"]
    env = ParkourEnv(
        level,
        EnvConfig(
            level_id=level.id,
            randomize_reset=False,
            max_episode_seconds=0.25,
            action_smoothing=0.0,
        ),
    )
    env.reset(seed=0)
    steps = 0
    while True:
        result = env.step(np.zeros(ACTION_DIM))
        steps += 1
        if result.terminated or result.truncated:
            break
    assert result.truncated and not result.terminated
    assert result.info["termination_reason"] == "timeout"
    assert steps == env.max_steps == 25
    env.close()


def test_leaving_the_course_terminates(env: ParkourEnv) -> None:
    env.reset(seed=0)
    env.data.qpos[0] = env.level.bounds_max.x + 1.0
    mujoco.mj_forward(env.model, env.data)
    result = env.step(np.zeros(ACTION_DIM))
    assert result.terminated
    assert result.info["termination_reason"] == "out_of_bounds"


def test_falling_into_the_void_terminates(catalog: LevelCatalog) -> None:
    level = catalog["level-16-rooftops"]
    env = ParkourEnv(level, EnvConfig(level_id=level.id, randomize_reset=False))
    env.reset(seed=0)
    env.data.qpos[2] = level.bounds_min.z + 0.2
    mujoco.mj_forward(env.model, env.data)
    result = env.step(np.zeros(ACTION_DIM))
    assert result.terminated
    assert result.info["termination_reason"] == "void_fall"
    env.close()


def test_episode_summary_is_serialisable(env: ParkourEnv) -> None:
    import json

    env.reset(seed=4)
    for _ in range(5):
        result = env.step(np.zeros(ACTION_DIM))
        if result.terminated or result.truncated:
            break
    summary = env.episode_summary()
    assert summary["level_id"] == "level-01-basics"
    assert summary["sim_steps"] == env.steps
    assert summary["success"] is False
    assert summary["termination_reason"] in (
        "fell_over",
        "collapsed",
        "none",
        "timeout",
        "course_complete",
    )
    assert 0.0 <= summary["completion_fraction"] <= 1.0
    assert isinstance(summary["metrics"], dict)
    json.dumps(summary)


def test_replay_frames_are_recorded_without_rendering(catalog: LevelCatalog) -> None:
    level = catalog["level-01-basics"]
    env = ParkourEnv(
        level,
        EnvConfig(
            level_id=level.id,
            seed=2,
            record_frames=True,
            frame_stride=2,
            randomize_reset=False,
        ),
    )
    env.reset(seed=2)
    for _ in range(20):
        result = env.step(np.zeros(ACTION_DIM))
        if result.terminated or result.truncated:
            break
    frames = env.frames
    assert frames
    frame = frames[-1]
    assert set(frame["bodies"]) >= {"pelvis", "torso", "head", "foot_l", "foot_r"}
    for values in frame["bodies"].values():
        assert len(values) == 7
        quaternion = np.array(values[3:])
        assert math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-3)
    assert frame["t"] == pytest.approx(env.steps * env.config.control_dt, abs=1e-9)
    env.close()


def test_render_frame_fails_cleanly_without_a_display(
    env: ParkourEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    with pytest.raises(RuntimeError, match="no display"):
        env.render_frame(16, 16)


def test_config_round_trip(catalog: LevelCatalog) -> None:
    config = EnvConfig(
        level_id="level-02-step-up",
        seed=12,
        action_scale=0.3,
        action_smoothing=0.4,
        actuation="position",
        reward=RewardConfig(progress_per_metre=6.0, fall_penalty=9.0),
    )
    data = config.to_dict()
    assert data["reward"]["progress_per_metre"] == 6.0
    assert data["action_scale"] == 0.3
    restored = RewardConfig.from_dict(data["reward"])
    assert restored.progress_per_metre == 6.0
    with pytest.raises(ValueError, match="unknown reward weights"):
        RewardConfig.from_dict({"nonsense": 1.0})
    del catalog


def test_moving_platform_motion_is_periodic(catalog: LevelCatalog) -> None:
    level = catalog["level-10-moving-bridge"]
    env = ParkourEnv(level, EnvConfig(level_id=level.id))
    env.reset(seed=0)
    spec = env.scene.moving_platforms[0]
    period = spec.period
    positions = [
        np.array(spec.position_at(t), dtype=float) for t in (0.0, period / 4, period / 2, period)
    ]
    start = positions[0]
    assert float(np.linalg.norm(positions[-1] - start)) == pytest.approx(0.0, abs=1e-6)
    assert max(float(np.linalg.norm(position - start)) for position in positions) > 0.01
    env.close()


def test_observation_blocks_do_not_overwrite_each_other(env: ParkourEnv) -> None:
    """Regression: the world-position triple used to be written at 0:3 and then clobbered."""

    observation, _ = env.reset(seed=5)
    env.step(np.zeros(ACTION_DIM))
    observation = env._build_observation()
    names = OBSERVATION_NAMES
    support = env._last_support
    pelvis_z = float(env.data.qpos[2])
    up_body = observation[2:5]
    assert observation[names.index("support_height")] == pytest.approx(
        min(max(pelvis_z - support, 0.0), 2.0), abs=1e-9
    )
    assert observation[names.index("pelvis_z")] == pytest.approx(min(max(pelvis_z / 4.0, -1.0), 2.0))
    # up_z and body_up_z are the same quantity and must agree.
    assert observation[names.index("up_z")] == pytest.approx(float(up_body[2]), abs=1e-9)
    linear = observation[5:8]
    angular = observation[8:11]
    assert np.all(np.abs(linear) <= 1.5 + 1e-9)
    assert np.all(np.abs(angular) <= 1.2 + 1e-9)
    assert np.count_nonzero(observation[:OBS_JOINT_POS_START]) >= 3


def test_gait_reward_only_counts_foot_exchanges(catalog: LevelCatalog) -> None:
    """The gait term rewards a clean left/right exchange, not hopping or standing."""

    level = catalog["level-01-basics"]
    env = ParkourEnv(level, EnvConfig(level_id=level.id, reward=RewardConfig(gait_per_second=5.0)))
    try:
        env.reset(seed=0)
        assert env._gait_event((True, True)) is False  # first observation: no baseline
        assert env._gait_event((True, True)) is False  # double support: nothing changes
        assert env._gait_event((True, False)) is False  # lifting the right foot is not a step
        assert env._gait_event((True, True)) is True  # right foot lands while left is planted
        assert env._gait_event((True, True)) is False  # no repeated credit for the same foot
        assert env._gait_event((False, True)) is False
        assert env._gait_event((True, True)) is True  # left foot lands

        # A hop (both feet leave and both land) is not an exchange.
        env._gait_event((False, False))
        assert env._gait_event((True, True)) is False

        # The reported component uses the configured weight and control step.
        env._previous_foot_state = (True, False)
        env.step(np.zeros(ACTION_DIM))
        components = {}
        for _ in range(60):
            result = env.step(np.zeros(ACTION_DIM))
            components = result.info["reward_components"]
        assert "gait" not in components or components["gait"] == pytest.approx(
            5.0 * env.config.control_dt
        )
    finally:
        env.close()


def test_gait_weight_defaults_to_off() -> None:
    assert RewardConfig().gait_per_second == 0.0
    assert RewardConfig.posture_stage().gait_per_second == 0.0


def test_stagnation_termination_ends_a_standing_episode(catalog: LevelCatalog) -> None:
    """Standing still must end the episode, otherwise 'do nothing' is a stable optimum."""

    level = catalog["level-01-basics"]
    env = ParkourEnv(
        level,
        EnvConfig(
            level_id=level.id,
            physics_timestep=0.002,
            control_decimation=5,
            max_episode_seconds=10.0,
            stagnation_seconds=1.0,
        ),
    )
    try:
        env.reset(seed=0)
        for step in range(500):
            result = env.step(np.zeros(ACTION_DIM))
            if result.terminated or result.truncated:
                break
        assert result.terminated is True
        assert result.truncated is False
        assert result.info["termination_reason"] == "no_progress"
        assert (step + 1) * env.config.control_dt == pytest.approx(1.0, abs=0.05)
    finally:
        env.close()

    # Off by default: the same standing policy survives until the level time limit.
    env = ParkourEnv(level, EnvConfig(level_id=level.id, max_episode_seconds=3.0))
    try:
        env.reset(seed=0)
        for step in range(500):
            result = env.step(np.zeros(ACTION_DIM))
            if result.terminated or result.truncated:
                break
        assert result.truncated is True
        assert result.info["termination_reason"] == "timeout"
    finally:
        env.close()


def test_stagnation_does_not_fire_while_approaching(catalog: LevelCatalog) -> None:
    level = catalog["level-01-basics"]
    env = ParkourEnv(
        level,
        EnvConfig(level_id=level.id, max_episode_seconds=10.0, stagnation_seconds=1.0),
    )
    try:
        env.reset(seed=0)
        # Teleport the robot forward each step: the goal distance keeps shrinking, so the
        # no-progress rule must stay quiet even though no physics is doing any work.
        for _ in range(150):
            env.data.qpos[0] += 0.05
            mujoco.mj_forward(env.model, env.data)
            result = env.step(np.zeros(ACTION_DIM))
            if result.terminated or result.truncated:
                break
        assert result.info["termination_reason"] != "no_progress"
    finally:
        env.close()


def test_config_timestep_is_authoritative(catalog: LevelCatalog) -> None:
    """Regression: the MJCF default timestep used to win over ``EnvConfig``.

    Every derived quantity (control_dt, episode length, time-based reward terms, stagnation
    timers) assumes the model really integrates at ``physics_timestep``.
    """

    level = catalog["level-01-basics"]
    for timestep, decimation in ((0.002, 5), (0.004, 5), (0.002, 10)):
        env = ParkourEnv(
            level,
            EnvConfig(
                level_id=level.id,
                physics_timestep=timestep,
                control_decimation=decimation,
                max_episode_seconds=1.0,
            ),
        )
        try:
            assert env.model.opt.timestep == pytest.approx(timestep)
            assert env.config.control_dt == pytest.approx(timestep * decimation)
            env.reset(seed=0)
            steps = 0
            while True:
                result = env.step(np.zeros(ACTION_DIM))
                steps += 1
                if result.terminated or result.truncated:
                    break
            assert result.info["termination_reason"] == "timeout"
            # The number of control steps matches one second of simulated time.
            assert steps * timestep * decimation == pytest.approx(1.0, abs=1e-9)
        finally:
            env.close()


def test_stagnation_timer_uses_simulated_seconds(catalog: LevelCatalog) -> None:
    level = catalog["level-01-basics"]
    env = ParkourEnv(
        level,
        EnvConfig(
            level_id=level.id,
            physics_timestep=0.004,
            control_decimation=5,
            max_episode_seconds=10.0,
            stagnation_seconds=1.5,
        ),
    )
    try:
        env.reset(seed=0)
        steps = 0
        while True:
            result = env.step(np.zeros(ACTION_DIM))
            steps += 1
            if result.terminated or result.truncated:
                break
        assert result.info["termination_reason"] == "no_progress"
        assert steps * 0.02 == pytest.approx(1.5, abs=0.05)
    finally:
        env.close()
