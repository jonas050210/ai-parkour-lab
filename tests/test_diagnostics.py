"""Diagnostics: does the policy stand, shuffle, hop, walk, and where does it end?"""

from __future__ import annotations

import numpy as np
import pytest

from parkour_lab.diagnostics import (
    EpisodeTrace,
    diagnose_policy,
    format_diagnosis,
    summarise_traces,
    trace_episode,
)
from parkour_lab.env import ACTION_DIM, EnvConfig
from parkour_lab.evaluation import rollout_episode
from parkour_lab.levels import LevelCatalog
from parkour_lab.policies import ActorCritic

LEVEL = "level-01-basics"


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


def _trace(**overrides) -> EpisodeTrace:
    values = dict(
        seed=11,
        level_id=LEVEL,
        success=False,
        termination_reason="collapsed",
        steps=150,
        seconds=3.0,
        completion=0.5,
        distance=3.0,
        max_distance=3.1,
        mean_forward_speed=1.0,
        peak_forward_speed=1.3,
        flight_fraction=0.05,
        contact_switches=6,
        final_pelvis_height=0.5,
        min_pelvis_height=0.45,
    )
    values.update(overrides)
    return EpisodeTrace(**values)


def _names(diagnosis: dict) -> set[str]:
    return {finding["name"] for finding in diagnosis["findings"]}


def test_trace_episode_agrees_with_rollout_episode(catalog: LevelCatalog) -> None:
    """The diagnostic path must be the same simulation as evaluation, not a lookalike."""

    policy = ActorCritic(90, ACTION_DIM, hidden_dim=16, seed=2)
    level = catalog[LEVEL]
    config = EnvConfig(level_id=level.id, max_episode_seconds=1.5, stagnation_seconds=0.0)
    record = rollout_episode(
        policy, level, seed=7, env_config=config, deterministic=True, randomize_reset=False
    )
    trace = trace_episode(
        policy, level, seed=7, env_config=config, deterministic=True, randomize_reset=False
    )
    assert trace.steps == record["sim_steps"]
    assert trace.success == record["success"]
    assert trace.termination_reason == record["termination_reason"]
    assert trace.completion == pytest.approx(record["completion_fraction"])
    assert trace.seconds == pytest.approx(record["elapsed_seconds"])


def test_diagnose_standing_policy_reports_no_progress(catalog: LevelCatalog) -> None:
    """A policy that never moves must be described as such, with evidence."""

    class StandingPolicy:
        def act(self, observation, *, deterministic=True, rng=None):
            return np.zeros(ACTION_DIM)

    level = catalog[LEVEL]
    config = EnvConfig(level_id=level.id, max_episode_seconds=2.0, stagnation_seconds=0.0)
    diagnosis = diagnose_policy(
        StandingPolicy(), level, seeds=(1, 2), env_config=config  # type: ignore[arg-type]
    )
    assert diagnosis["episodes"] == 2
    assert diagnosis["successes"] == 0
    assert diagnosis["mean_distance"] == pytest.approx(0.0, abs=0.15)
    assert diagnosis["flight_fraction"] == pytest.approx(0.0, abs=0.05)
    names = _names(diagnosis)
    assert "no_success" in names
    assert "shuffling" in names
    assert "stuck_near_start" in names
    assert "hopping" not in names
    text = format_diagnosis(diagnosis)
    assert "0/2 episodes complete" in text
    assert "mostly stands" in text or "shuffles" in text


def test_summarise_flags_hopping_and_missing_weight_transfer() -> None:
    hopping = summarise_traces(
        [_trace(flight_fraction=0.35, contact_switches=0, completion=0.4, distance=2.4)] * 3
    )
    assert "hopping" in _names(hopping)
    assert "no_weight_transfer" in _names(hopping)
    assert "shuffling" not in _names(hopping)
    assert hopping["flight_fraction"] == pytest.approx(0.35)

    shuffling = summarise_traces(
        [_trace(flight_fraction=0.0, mean_forward_speed=0.1, completion=0.05, distance=0.3)]
    )
    assert "shuffling" in _names(shuffling)
    assert "stuck_near_start" in _names(shuffling)
    assert "hopping" not in _names(shuffling)


def test_summarise_reports_success_and_seed_spread() -> None:
    success = summarise_traces(
        [
            _trace(success=True, termination_reason="course_complete", completion=1.0, distance=6.0),
            _trace(success=True, termination_reason="course_complete", completion=1.0, distance=6.0),
        ]
    )
    assert "success" in _names(success)
    assert success["success_rate"] == pytest.approx(1.0)
    assert "no_success" not in _names(success)

    spread = summarise_traces(
        [
            _trace(seed=1, completion=0.6),
            _trace(seed=2, completion=0.6),
            _trace(seed=3, completion=0.05, distance=0.2),
        ]
    )
    assert "seed_sensitivity" in _names(spread)
    assert spread["completion_spread"] == pytest.approx(0.55)

    with pytest.raises(ValueError):
        summarise_traces([])


def test_diagnose_uses_every_seed(catalog: LevelCatalog) -> None:
    """``env_config_from_checkpoint`` no longer pins ``randomize_reset=False``."""

    policy = ActorCritic(90, ACTION_DIM, hidden_dim=16, seed=1)
    level = catalog[LEVEL]
    config = EnvConfig(level_id=level.id, max_episode_seconds=1.5, randomize_reset=True)
    diagnosis = diagnose_policy(policy, level, seeds=(11, 23, 37), env_config=config)
    seeds = [trace["seed"] for trace in diagnosis["traces"]]
    assert seeds == [11, 23, 37]
    # Different reset noise per seed: the episodes are not three copies of one rollout.
    assert len({trace["completion"] for trace in diagnosis["traces"]}) > 1
    assert diagnosis["randomize_reset"] is True
