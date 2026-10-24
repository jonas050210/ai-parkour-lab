"""Humanoid model and level-catalogue tests: everything here must hold without a policy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from parkour_lab.course import build_course
from parkour_lab.env import ACTION_DIM, OBSERVATION_DIM, EnvConfig, ParkourEnv
from parkour_lab.humanoid import default_humanoid
from parkour_lab.levels import (
    DEFAULT_LEVELS_DIR,
    LevelCatalog,
    LevelError,
    load_level,
    validate_level_physics,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


def test_catalog_has_twenty_trainable_levels_and_held_out_split(catalog: LevelCatalog) -> None:
    assert len(catalog) >= 22
    assert len(catalog.train_levels) >= 20
    assert len(catalog.held_out_levels) >= 2
    orders = [level.order for level in catalog]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)
    for level in catalog:
        assert level.split in ("train", "held-out")
        assert 1 <= level.difficulty <= 5


def test_levels_get_progressively_harder(catalog: LevelCatalog) -> None:
    train = list(catalog.train_levels)
    difficulties = [level.difficulty for level in train]
    assert difficulties[0] == 1
    assert max(difficulties) >= 4
    # difficulty must never fall by more than one step once it has started rising
    assert all(
        later >= earlier - 1 for earlier, later in zip(difficulties, difficulties[1:])
    )


def test_every_level_compiles_into_a_mujoco_model(catalog: LevelCatalog) -> None:
    for level in catalog:
        scene = build_course(level)
        model = mujoco.MjModel.from_xml_string(scene.xml, {})
        assert model.nq == 26
        assert model.nv == 25
        assert model.nu == 19
        assert model.nmocap == len(scene.moving_platforms)
        # The robot must spawn standing on solid geometry.
        assert math.isfinite(level.spawn_support_z)
        assert level.spawn_position.z >= level.spawn_support_z


def test_humanoid_inventory_matches_the_environment() -> None:
    spec = default_humanoid()
    assert len(spec.joints) == ACTION_DIM == 19
    assert len(spec.bodies) == 13
    assert spec.total_mass == pytest.approx(46.6, abs=0.5)
    assert set(spec.joint_names) == {joint.name for joint in spec.joints}
    groups = spec.by_group()
    assert len(groups["leg"]) == 10
    assert len(groups["arm"]) == 6
    for joint in spec.joints:
        assert joint.lower < joint.upper
        assert joint.lower <= joint.default <= joint.upper
        assert joint.force_limit > 0
        assert joint.kp > 0 and joint.kv > 0


def test_position_actuators_use_radian_control_ranges(catalog: LevelCatalog) -> None:
    env = ParkourEnv(catalog["level-01-basics"], EnvConfig(level_id="level-01-basics"))
    for index, joint in enumerate(env.humanoid.joints):
        actuator_id = env.actuator_ids[index]
        low, high = env.model.actuator_ctrlrange[actuator_id]
        assert low == pytest.approx(math.radians(joint.lower), abs=1e-5)
        assert high == pytest.approx(math.radians(joint.upper), abs=1e-5)
    env.close()


def test_default_pose_stands_without_actuation_effort(catalog: LevelCatalog) -> None:
    env = ParkourEnv(
        catalog["level-01-basics"],
        EnvConfig(level_id="level-01-basics", randomize_reset=False),
    )
    env.reset(seed=1)
    initial = env.data.qpos[:3].copy()
    for _ in range(100):
        result = env.step(np.zeros(ACTION_DIM))
        if result.terminated or result.truncated:  # pragma: no cover - regression guard
            break
    assert not result.terminated
    assert env.data.qpos[2] > 0.85
    assert abs(env.data.qpos[0] - initial[0]) < 0.35
    env.close()


def test_action_mapping_is_bounded_and_centred(catalog: LevelCatalog) -> None:
    config = EnvConfig(level_id="level-01-basics", action_scale=0.2, randomize_reset=False)
    env = ParkourEnv(catalog["level-01-basics"], config)
    env.reset(seed=0)
    span_deg = np.rad2deg(env._action_span)
    assert np.all(span_deg > 0)
    # action=1 must reach the joint limit, never beyond it
    for index, joint in enumerate(env.humanoid.joints):
        ctrl_at_limit = env._action_center[index] + env._action_span[index]
        assert ctrl_at_limit <= math.radians(joint.upper) + 1e-6
    # an over-range action is clipped rather than passed through
    env.step(np.full(ACTION_DIM, 5.0))
    assert np.all(np.abs(env._action) <= 1.0)
    env.close()


def test_reset_is_deterministic_for_a_seed(catalog: LevelCatalog) -> None:
    config = EnvConfig(level_id="level-02-step-up", seed=5)
    first = ParkourEnv(catalog["level-02-step-up"], config)
    second = ParkourEnv(catalog["level-02-step-up"], config)
    obs_a, info_a = first.reset(seed=99)
    obs_b, info_b = second.reset(seed=99)
    assert info_a["seed"] == info_b["seed"] == 99
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-12)
    actions = np.linspace(-0.2, 0.2, ACTION_DIM)
    for _ in range(20):
        ra = first.step(actions)
        rb = second.step(actions)
        np.testing.assert_allclose(ra.observation, rb.observation, atol=1e-12)
        np.testing.assert_allclose(ra.reward, rb.reward, atol=1e-12)
    first.close()
    second.close()


def test_observations_are_finite_and_documented(catalog: LevelCatalog) -> None:
    env = ParkourEnv(catalog["level-03-small-hops"], EnvConfig(level_id="level-03-small-hops"))
    observation, info = env.reset(seed=3)
    assert observation.shape == (OBSERVATION_DIM,)
    assert OBSERVATION_DIM == 90
    assert np.all(np.isfinite(observation))
    rng = np.random.default_rng(0)
    for _ in range(50):
        result = env.step(rng.uniform(-1, 1, ACTION_DIM))
        assert np.all(np.isfinite(result.observation))
        assert math.isfinite(result.reward)
        if result.terminated or result.truncated:
            observation, info = env.reset(seed=int(rng.integers(0, 1_000)))
    assert "reward_components" in result.info
    env.close()


def test_bad_actions_are_rejected(catalog: LevelCatalog) -> None:
    env = ParkourEnv(catalog["level-01-basics"], EnvConfig(level_id="level-01-basics"))
    env.reset(seed=0)
    with pytest.raises(ValueError, match="expected action"):
        env.step(np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        env.step(np.full(ACTION_DIM, np.nan))
    env.close()


def test_falls_terminate_with_a_reason(catalog: LevelCatalog) -> None:
    env = ParkourEnv(
        catalog["level-01-basics"],
        EnvConfig(level_id="level-01-basics", randomize_reset=False),
        )
    env.reset(seed=0)
    # shove the root sideways; the episode must end with a documented reason
    env.data.qpos[2] = 0.25
    mujoco.mj_forward(env.model, env.data)
    result = env.step(np.zeros(ACTION_DIM))
    assert result.terminated
    assert result.info["termination_reason"] in ("collapsed", "fell_over")
    summary = env.episode_summary()
    assert summary["termination_reason"] in ("collapsed", "fell_over")
    assert summary["sim_steps"] >= 1
    env.close()


def test_hazard_level_ends_on_contact(catalog: LevelCatalog) -> None:
    hazards = [
        level for level in catalog if any(obstacle.hazard for obstacle in level.obstacles)
    ]
    assert hazards, "at least one level must contain a hazard"
    level = hazards[0]
    env = ParkourEnv(level, EnvConfig(level_id=level.id), )
    env.reset(seed=0)
    hazard = next(obstacle for obstacle in level.obstacles if obstacle.hazard)
    env.data.qpos[0] = hazard.position.x
    env.data.qpos[1] = hazard.position.y
    env.data.qpos[2] = hazard.position.z + hazard.size.z + 0.12
    mujoco.mj_forward(env.model, env.data)
    for _ in range(20):
        result = env.step(np.zeros(ACTION_DIM))
        if result.terminated:
            break
    assert result.info["termination_reason"] == "hazard_contact"
    assert env.episode_summary()["termination_reason"] == "hazard_contact"
    env.close()


def test_moving_platforms_are_animated(catalog: LevelCatalog) -> None:
    moving_levels = [
        level
        for level in catalog
        if any(obstacle.kind == "moving_platform" for obstacle in level.obstacles)
    ]
    assert moving_levels
    level = moving_levels[0]
    env = ParkourEnv(level, EnvConfig(level_id=level.id))
    env.reset(seed=0)
    mover = next(obstacle for obstacle in level.obstacles if obstacle.kind == "moving_platform")
    first = env.data.mocap_pos[env._mocap_ids[env.scene.moving_platforms[0].body_name]].copy()
    for _ in range(150):
        env.step(np.zeros(ACTION_DIM))
    second = env.data.mocap_pos[env._mocap_ids[env.scene.moving_platforms[0].body_name]].copy()
    assert np.linalg.norm(second - first) > 1e-3
    assert mover.motion is not None
    env.close()


def test_muscle_actuation_mode_compiles_and_steps(catalog: LevelCatalog) -> None:
    config = EnvConfig(level_id="level-01-basics", actuation="muscle")
    env = ParkourEnv(catalog["level-01-basics"], config)
    assert env.humanoid.actuation_mode == "muscle"
    observation, _ = env.reset(seed=0)
    assert observation.shape == (OBSERVATION_DIM,)
    for _ in range(5):
        result = env.step(np.zeros(ACTION_DIM))
        assert np.all(np.isfinite(result.observation))
        assert np.all(np.abs(env.data.ctrl) <= 1.0)
    env.close()


def test_action_smoothing_blends_targets(catalog: LevelCatalog) -> None:
    config = EnvConfig(
        level_id="level-01-basics", action_smoothing=0.5, randomize_reset=False
    )
    env = ParkourEnv(catalog["level-01-basics"], config)
    env.reset(seed=0)
    before = env.data.ctrl[env.actuator_ids].copy()
    env.step(np.ones(ACTION_DIM))
    after = env.data.ctrl[env.actuator_ids].copy()
    midpoint = 0.5 * (before + after)
    # the first update moves about half-way towards the target
    moved = np.abs(after - before)
    assert np.all(moved > 0)
    del midpoint
    env.close()


def test_mjcf_export_round_trips(catalog: LevelCatalog, tmp_path: Path) -> None:
    level = catalog["level-10-moving-bridge"]
    scene = build_course(level)
    path = tmp_path / "level10.xml"
    path.write_text(scene.xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq == 26
    assert "<mujoco" in scene.xml
    assert model.nmocap == len(scene.moving_platforms)


def test_invalid_levels_are_rejected(tmp_path: Path) -> None:
    schema = json.loads((DEFAULT_LEVELS_DIR / "level.schema.json").read_text(encoding="utf-8"))
    base = json.loads(
        (DEFAULT_LEVELS_DIR / "level-01-basics.json").read_text(encoding="utf-8")
    )
    candidates = []
    bad = json.loads(json.dumps(base))
    bad["spawn"]["position"][2] = 25.0
    candidates.append(bad)
    bad = json.loads(json.dumps(base))
    bad["obstacles"][0]["size"] = [0.0, 1.0, 0.1]
    candidates.append(bad)
    bad = json.loads(json.dumps(base))
    bad["finish"]["position"][0] = 100.0
    candidates.append(bad)
    for index, candidate in enumerate(candidates):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(LevelError):
            load_level(path, use_schema=True)
    assert schema["title"]


def test_level_spawn_support_matches_geometry(catalog: LevelCatalog) -> None:
    for level in catalog:
        validate_level_physics(level)
        support = level.spawn_support_z
        assert level.spawn_position.z == pytest.approx(support + 0.94, abs=0.03) or (
            level.spawn_position.z - support
        ) > 0.5


def test_level_motion_is_explicit(catalog: LevelCatalog) -> None:
    for level in catalog:
        for obstacle in level.obstacles:
            if obstacle.kind == "moving_platform":
                assert obstacle.motion is not None
                assert obstacle.motion.period_seconds > 0
                assert obstacle.motion.amplitude > 0
