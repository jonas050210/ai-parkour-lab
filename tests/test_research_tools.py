"""Tests for the secondary research tools: bounded evolution and Harness².

Both tools are deliberately constrained.  The tests below pin the constraints, not just the
happy path: out-of-range proposals must be clamped, off-surface proposals must be rejected,
and neither tool may modify a saved run or silently replace PPO.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from parkour_lab.harness2 import (
    FORBIDDEN_PREFIXES,
    KNOB_BOUNDS,
    AssistantConfig,
    ProposalError,
    ResearchAssistant,
    validate_changes,
)
from parkour_lab.levels import LevelCatalog
from parkour_lab.ppo import PPOConfig
from parkour_lab.storage import RunStore
from parkour_lab.training.evolution import (
    PARAMETERS,
    EvolutionConfig,
    apply_overrides,
    default_vector,
    evolve_reward_weights,
    vector_to_overrides,
)
from parkour_lab.training.runner import TrainConfig, train


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


@pytest.fixture(scope="module")
def base_config(tmp_path_factory: pytest.TempPathFactory) -> TrainConfig:
    return TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        seed=4,
        total_steps=1024,
        rollout_steps=128,
        num_envs=2,
        ppo=PPOConfig(epochs=2, minibatch_size=64),
        eval_interval=2,
        eval_levels=("level-01-basics",),
        eval_seeds=(11, 23),
        checkpoint_interval=2,
        keep_checkpoints=2,
        hidden_dim=20,
        max_replays=1,
        max_episode_seconds=3.0,
    )


@pytest.fixture(scope="module")
def research_run(
    catalog: LevelCatalog, base_config: TrainConfig, tmp_path_factory: pytest.TempPathFactory
) -> RunStore:
    """One real, short run that the harness and evolution tests can inspect."""

    runs_dir = tmp_path_factory.mktemp("research-runs")
    train(base_config, catalog, runs_dir=runs_dir, run_id="research-base")
    return RunStore.load(runs_dir / "research-base")


# --------------------------------------------------------------------------------------
# Evolution
# --------------------------------------------------------------------------------------
def test_default_vector_reproduces_the_defaults() -> None:
    overrides = vector_to_overrides(default_vector())
    assert overrides["reward.progress_per_metre"] == pytest.approx(5.0, rel=1e-6)
    assert overrides["ppo.learning_rate"] == pytest.approx(PPOConfig().learning_rate, rel=0.15)
    assert overrides["action_scale"] == pytest.approx(0.25, rel=1e-6)


def test_parameters_round_trip_through_unit_space() -> None:
    rng = np.random.default_rng(0)
    for parameter in PARAMETERS:
        for unit in rng.uniform(0, 1, 5):
            value = parameter.from_unit(float(unit))
            assert parameter.low <= value <= parameter.high
            assert parameter.to_unit(value) == pytest.approx(float(unit), abs=1e-9)
        # Out-of-range inputs are clamped, never extrapolated.
        assert parameter.from_unit(-5.0) == pytest.approx(parameter.low)
        assert parameter.from_unit(5.0) == pytest.approx(parameter.high)


def test_vector_length_is_validated() -> None:
    with pytest.raises(ValueError, match="expected"):
        vector_to_overrides([0.5])


def test_apply_overrides_only_touches_evolvable_fields(base_config: TrainConfig) -> None:
    updated = apply_overrides(
        base_config,
        {
            "reward.progress_per_metre": 9.0,
            "ppo.learning_rate": 8e-4,
            "action_scale": 0.33,
        },
    )
    assert updated.reward.progress_per_metre == pytest.approx(9.0)
    assert updated.ppo.learning_rate == pytest.approx(8e-4)
    assert updated.action_scale == pytest.approx(0.33)
    # Physics, levels and the learning algorithm itself are untouched.
    assert updated.physics_timestep == base_config.physics_timestep
    assert updated.control_decimation == base_config.control_decimation
    assert updated.level_ids == base_config.level_ids
    assert updated.ppo.gamma == base_config.ppo.gamma
    assert updated.reward.fall_penalty == base_config.reward.fall_penalty


def test_apply_overrides_rejects_unknown_knobs(base_config: TrainConfig) -> None:
    with pytest.raises(ValueError, match="unknown evolvable parameters"):
        apply_overrides(base_config, {"physics.timestep": 0.01})


def test_evolution_runs_a_tiny_search(
    catalog: LevelCatalog, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = EvolutionConfig(
        generations=1,
        population=2,
        train_steps=512,
        num_envs=1,
        workers=1,
        seed=0,
        eval_seeds=(11,),
    )
    base = TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        rollout_steps=128,
        num_envs=1,
        ppo=PPOConfig(epochs=2, minibatch_size=64),
        hidden_dim=16,
        max_episode_seconds=3.0,
    )
    messages: list[str] = []
    result = evolve_reward_weights(
        catalog,
        ["level-01-basics"],
        config=config,
        runs_dir=tmp_path,
        base_config=base,
        logger=messages.append,
    )
    summary = result["summary"]
    assert summary["generations"] == 1
    assert summary["population"] == 2
    assert summary["best_run_id"]
    assert len(summary["candidates"]) == 2
    assert "never replaces the learning algorithm" in summary["note"]
    assert all(candidate["run_id"] for candidate in summary["candidates"])

    run_dir = Path(result["run_dir"])
    payload = json.loads((run_dir / "result.json").read_text())
    assert payload["best_overrides"] == summary["best_overrides"]
    history = (run_dir / "evolution.jsonl").read_text().strip().splitlines()
    assert len(history) == 2
    row = json.loads(history[0])
    assert set(row["overrides"]) == {parameter.name for parameter in PARAMETERS}

    # Every candidate really trained and saved a run we can reload.
    for candidate in summary["candidates"]:
        store = RunStore.load(tmp_path / candidate["run_id"])
        assert store.read_summary()["status"] == "completed"


# --------------------------------------------------------------------------------------
# Harness²
# --------------------------------------------------------------------------------------
def test_validate_changes_clamps_to_the_knob_range() -> None:
    cleaned = validate_changes(
        {"reward.progress_per_metre": 999.0, "ppo.gamma": 0.1, "action_scale": 0.2}
    )
    assert cleaned["reward.progress_per_metre"] == KNOB_BOUNDS["reward.progress_per_metre"][1]
    assert cleaned["ppo.gamma"] == KNOB_BOUNDS["ppo.gamma"][0]
    assert cleaned["action_scale"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "changes",
    [
        {"physics.timestep": 0.01},
        {"humanoid.total_mass": 80.0},
        {"level-01-basics.finish_x": 1.0},
        {"course.gravity": -1.0},
        {"observation.noise": 0.1},
        {"checkpoint.path": "/tmp/x.npz"},
        {"reward.not_a_weight": 1.0},
        {"reward.progress_per_metre": float("nan")},
        {"reward.progress_per_metre": "lots"},
        {},
        {1: 0.5},
    ],
)
def test_validate_changes_rejects_off_surface_edits(changes: dict) -> None:
    with pytest.raises(ProposalError):
        validate_changes(changes)


def test_forbidden_prefixes_cover_the_simulation_surface() -> None:
    for prefix in ("humanoid", "level", "course", "physics", "observation"):
        assert prefix in FORBIDDEN_PREFIXES


def test_plan_without_a_run_returns_bounded_advice(catalog: LevelCatalog) -> None:
    assistant = ResearchAssistant(catalog, AssistantConfig(max_proposals=2))
    proposals = assistant.plan()
    assert 1 <= len(proposals) <= 2
    for proposal in proposals:
        assert proposal.changes
        assert set(proposal.changes) <= set(KNOB_BOUNDS)
        payload = proposal.to_dict()
        assert payload["applied"] is False
        assert "parkour train" in payload["how_to_apply"]


def test_plan_reads_a_real_run_and_stays_in_bounds(
    catalog: LevelCatalog, research_run: RunStore
) -> None:
    assistant = ResearchAssistant(catalog, AssistantConfig(min_improvement=0.02))
    proposals = assistant.plan(run_dir=research_run.paths.root)
    assert proposals, "a stalled 0% run should produce at least one suggestion"
    for proposal in proposals:
        assert set(proposal.changes) <= set(KNOB_BOUNDS)
        for name, value in proposal.changes.items():
            low, high = KNOB_BOUNDS[name]
            assert low <= value <= high
        assert proposal.rationale
        assert proposal.evidence["run_id"] == "research-base"

    before = (research_run.paths.root / "config.json").read_text()
    assistant.plan(run_dir=research_run.paths.root)
    # Planning is read-only: it must never rewrite the run it inspected.
    assert (research_run.paths.root / "config.json").read_text() == before


def test_plan_rejects_an_unknown_level(catalog: LevelCatalog) -> None:
    assistant = ResearchAssistant(catalog)
    with pytest.raises(Exception):
        assistant.plan(target_levels=["level-99-missing"])


def test_probe_writes_a_report_for_a_real_checkpoint(
    catalog: LevelCatalog, research_run: RunStore
) -> None:
    assistant = ResearchAssistant(
        catalog,
        AssistantConfig(probe_levels=("level-01-basics", "level-02-step-up"), probe_seeds=(11,)),
    )
    report = assistant.probe(research_run.paths.root)
    assert report.ok is True
    assert report.checkpoint.endswith(".npz")
    assert len(report.levels) == 2
    assert report.aggregate["episodes"] == 2
    assert set(report.weak_levels) <= {"level-01-basics", "level-02-step-up"}
    path = research_run.paths.root / "harness2"
    assert path.is_dir()
    payload = json.loads(next(path.glob("probe-*.json")).read_text())
    assert payload["restrictions"]["physics"] == "untouched"
    assert payload["restrictions"]["controller"] == "untouched"
    assert payload["restrictions"]["learning_algorithm"] == "untouched"
    # The report records the checkpoint's own dynamics, not this process's defaults.
    assert payload["config_action_scale"] == pytest.approx(
        research_run.read_config()["action_scale"]
    )


def test_probe_requires_a_checkpoint(catalog: LevelCatalog, tmp_path) -> None:
    store = RunStore.create(
        tmp_path,
        run_id="empty-run",
        config={"level_ids": ["level-01-basics"], "reward": {}, "ppo": {}},
    )
    del store
    assistant = ResearchAssistant(catalog, AssistantConfig(probe_levels=("level-01-basics",)))
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        assistant.probe(tmp_path / "empty-run")


def test_network_proposals_are_off_by_default(catalog: LevelCatalog) -> None:
    assistant = ResearchAssistant(catalog, AssistantConfig(allow_network=False))
    assert assistant.config.allow_network is False
    assert assistant.config.endpoint is None
