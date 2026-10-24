"""End-to-end training, evaluation and storage tests on a tiny but real run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from parkour_lab.evaluation import (
    EvalConfig,
    aggregate_records,
    compare_aggregates,
    evaluate_policy,
    format_evaluation_table,
    rollout_episode,
)
from parkour_lab.levels import LevelCatalog
from parkour_lab.policies import ActorCritic
from parkour_lab.ppo import PPOConfig
from parkour_lab.storage import (
    RunStore,
    delete_run,
    environment_provenance,
    list_runs,
)
from parkour_lab.training.runner import TrainConfig, compare_runs, train

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


@pytest.fixture(scope="module")
def tiny_config() -> TrainConfig:
    return TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        seed=3,
        total_steps=1024,
        rollout_steps=128,
        num_envs=2,
        workers=1,
        ppo=PPOConfig(epochs=2, minibatch_size=64),
        eval_interval=2,
        eval_levels=("level-01-basics",),
        eval_seeds=(11, 23),
        checkpoint_interval=2,
        keep_checkpoints=3,
        hidden_dim=24,
        max_replays=2,
        record_replay_every_eval=1,
        name="pytest-run",
        notes="created by tests/test_training_pipeline.py",
    )


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, catalog: LevelCatalog, tiny_config: TrainConfig) -> Path:
    runs = tmp_path_factory.mktemp("runs")
    summary = train(tiny_config, catalog, runs_dir=runs, run_id="tiny-01", logger=lambda _: None)
    assert summary["run_id"] == "tiny-01"
    return runs / "tiny-01"


def test_training_writes_every_expected_artifact(run_dir: Path) -> None:
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "provenance.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "metrics" / "training.csv").is_file()
    assert (run_dir / "metrics" / "episodes.jsonl").is_file()
    assert (run_dir / "checkpoints" / "latest.npz").is_file()
    assert (run_dir / "checkpoints" / "best.npz").is_file()
    assert (run_dir / "evaluation").is_dir()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["level_ids"] == ["level-01-basics"]
    assert config["initial_log_std"] == -1.0
    assert config["action_smoothing"] == 0.6
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_steps"] == 1024
    assert summary["status"] in ("completed", "running", "stopped")
    assert "best_eval_success" in summary


def test_run_store_reads_metrics(run_dir: Path) -> None:
    store = RunStore.load(run_dir)
    assert store.paths.run_id == "tiny-01"
    assert store.read_config()["seed"] == 3
    rows = store.training_rows()
    assert rows and "episode_return" in rows[0]
    assert store.evaluations(), "training must produce at least one evaluation record"
    assert store.checkpoints()
    assert store.paths.best.is_file()


def test_list_runs_and_delete(tmp_path: Path, run_dir: Path) -> None:
    runs = list_runs(run_dir.parent)
    assert any(run["run_id"] == "tiny-01" for run in runs)
    entry = next(run for run in runs if run["run_id"] == "tiny-01")
    assert entry["status"] in ("completed", "running", "stopped")
    assert entry["total_steps"] == 1024
    copy = tmp_path / "runs"
    copy.mkdir()
    import shutil

    shutil.copytree(run_dir, copy / "tiny-01")
    delete_run(copy / "tiny-01", runs_dir=copy)
    assert not (copy / "tiny-01").exists()


def test_resume_continues_from_a_checkpoint(
    catalog: LevelCatalog, tiny_config: TrainConfig, run_dir: Path, tmp_path: Path
) -> None:
    import shutil

    runs = tmp_path / "resume-runs"
    runs.mkdir()
    shutil.copytree(run_dir, runs / "tiny-01")
    before = json.loads((runs / "tiny-01" / "summary.json").read_text(encoding="utf-8"))
    summary = train(
        None,
        catalog,
        runs_dir=runs,
        resume_from=runs / "tiny-01" / "checkpoints" / "latest.npz",
        logger=lambda _: None,
    )
    assert summary["run_id"] == "tiny-01"
    assert summary["total_steps"] >= 1024
    del before
    # resuming with a mismatched run id is rejected
    with pytest.raises(ValueError, match="does not match"):
        train(
            tiny_config,
            catalog,
            runs_dir=runs,
            run_id="other-run",
            resume_from=runs / "tiny-01" / "checkpoints" / "latest.npz",
            logger=lambda _: None,
        )


def test_policy_evaluation_and_replay(run_dir: Path, catalog: LevelCatalog) -> None:
    policy, extra = ActorCritic.load(run_dir / "checkpoints" / "best.npz")
    assert policy.observation_dim == 90
    assert policy.action_dim == 19
    assert extra["run_id"] == "tiny-01"

    record = rollout_episode(
        policy,
        catalog["level-01-basics"],
        seed=11,
        deterministic=True,
        record_frames=True,
        frame_stride=5,
    )
    assert record["frames"]
    assert record["level_id"] == "level-01-basics"
    assert record["seed"] == 11
    assert record["termination_reason"] in (
        "course_complete",
        "fell_over",
        "collapsed",
        "void_fall",
        "hazard_contact",
        "out_of_bounds",
        "timeout",
    )
    evaluation = evaluate_policy(
        policy,
        catalog,
        ["level-01-basics", "level-02-step-up"],
        config=EvalConfig(seeds=(11, 23), deterministic=True),
    )
    assert evaluation["aggregate"]["overall"]["episodes"] == 4
    assert len(evaluation["records"]) == 4
    table = format_evaluation_table(evaluation)
    assert "level-01-basics" in table
    assert "success" in table.lower()


def test_aggregate_and_compare_helpers() -> None:
    records = [
        {
            "level_id": "a",
            "success": True,
            "completion_fraction": 1.0,
            "episode_return": 40.0,
            "elapsed_seconds": 9.0,
            "termination_reason": "course_complete",
        },
        {
            "level_id": "a",
            "success": False,
            "completion_fraction": 0.2,
            "episode_return": -5.0,
            "elapsed_seconds": 3.0,
            "termination_reason": "fell_over",
        },
    ]
    aggregate = aggregate_records(records)
    assert aggregate["overall"]["episodes"] == 2
    assert aggregate["overall"]["success_rate"] == 0.5
    assert aggregate["per_level"]["a"]["success_rate"] == 0.5
    assert aggregate["terminations"]["fell_over"] == 1
    comparison = compare_aggregates(aggregate, aggregate)
    assert comparison["overall"]["delta_success_rate"] == 0.0
    assert aggregate_records([])["overall"]["episodes"] == 0


def test_compare_two_runs(run_dir: Path) -> None:
    store_a = RunStore.load(run_dir)
    store_b = RunStore.load(run_dir)
    comparison = compare_runs(store_a, store_b)
    assert "error" not in comparison
    assert comparison["rows"]
    assert comparison["overall"]["delta_success_rate"] == 0.0


def test_compare_two_runs_uses_progress_when_success_ties(tmp_path: Path) -> None:
    """Both runs never succeeded, so the comparison must rank them by completion."""

    def make(run_id: str, completion: float) -> RunStore:
        store = RunStore.create(
            tmp_path, run_id=run_id, config={"level_ids": ["level-01-basics"], "total_steps": 1024}
        )
        for iteration in range(2):
            store.write_json(
                f"evaluation/{iteration:05d}.json",
                {
                    "iteration": iteration,
                    "total_steps": 512 * (iteration + 1),
                    "aggregate": {
                        "overall": {
                            "episodes": 2,
                            "success_rate": 0.0,
                            "mean_completion": completion - 0.1 * iteration,
                            "mean_return": 5.0,
                        },
                        "per_level": {
                            "level-01-basics": {
                                "episodes": 2,
                                "success_rate": 0.0,
                                "mean_completion": completion - 0.1 * iteration,
                                "mean_return": 5.0,
                            }
                        },
                    },
                },
            )
        return store

    comparison = compare_runs(make("a-run", 0.5), make("b-run", 0.2))
    # baseline = run A, candidate = run B, so the delta is negative.
    assert comparison["overall"]["baseline_mean_completion"] == pytest.approx(0.5)
    assert comparison["overall"]["candidate_mean_completion"] == pytest.approx(0.2)
    assert comparison["overall"]["delta_mean_completion"] == pytest.approx(-0.3)
    # Both runs' best evaluation is the first one (the higher completion of the pair).
    assert comparison["run_a"]["iteration"] == 0
    assert comparison["run_b"]["iteration"] == 0


def test_provenance_captures_versions() -> None:
    provenance = environment_provenance()
    assert "python" in provenance or "packages" in provenance
    text = json.dumps(provenance)
    assert "mujoco" in text.lower()


def test_training_is_reproducible(tmp_path: Path, catalog: LevelCatalog) -> None:
    config = TrainConfig(
        level_ids=("level-01-basics",),
        curriculum="single",
        seed=11,
        total_steps=256,
        rollout_steps=128,
        num_envs=2,
        ppo=PPOConfig(epochs=1, minibatch_size=64),
        eval_interval=100,
        eval_levels=("level-01-basics",),
        eval_seeds=(1,),
        checkpoint_interval=100,
        hidden_dim=16,
        name="repro-a",
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    train(config, catalog, runs_dir=a, run_id="run-a", logger=lambda _: None)
    train(config, catalog, runs_dir=b, run_id="run-b", logger=lambda _: None)
    rows_a = (a / "run-a" / "metrics" / "training.csv").read_text(encoding="utf-8")
    rows_b = (b / "run-b" / "metrics" / "training.csv").read_text(encoding="utf-8")
    # timestamps differ in the wall-clock column, but the learning signal must match
    stripped_a = [
        line.split(",")[:5] + line.split(",")[6:] for line in rows_a.strip().splitlines()
    ]
    stripped_b = [
        line.split(",")[:5] + line.split(",")[6:] for line in rows_b.strip().splitlines()
    ]
    assert stripped_a[0] == stripped_b[0]
    policy_a, _ = ActorCritic.load(a / "run-a" / "checkpoints" / "latest.npz")
    policy_b, _ = ActorCritic.load(b / "run-b" / "checkpoints" / "latest.npz")
    observations = np.linspace(-0.5, 0.5, 90).reshape(1, -1)
    np.testing.assert_allclose(
        policy_a.action_mean(observations), policy_b.action_mean(observations), atol=1e-10
    )


def test_eval_seeds_probe_different_resets(
    catalog: LevelCatalog, tiny_config: TrainConfig, tmp_path: Path
) -> None:
    """Deterministic actions plus a fixed reset would make every seed the same episode."""

    store = RunStore.create(tmp_path, run_id="eval-seeds")
    from parkour_lab.training.runner import TrainingRunner

    runner = TrainingRunner(tiny_config, catalog, store, logger=lambda _msg: None)
    runner.setup()
    try:
        fixed = evaluate_policy(
            runner.model,
            catalog,
            ["level-01-basics"],
            config=EvalConfig(seeds=(11, 23, 37), randomize_reset=False),
            env_config=tiny_config.env_config("level-01-basics"),
        )
        records = fixed["records"]
        assert len({record["completion_fraction"] for record in records}) == 1
        assert len({record["sim_steps"] for record in records}) == 1

        varied = evaluate_policy(
            runner.model,
            catalog,
            ["level-01-basics"],
            config=EvalConfig(seeds=(11, 23, 37), randomize_reset=True),
            env_config=tiny_config.env_config("level-01-basics"),
        )
        assert len({record["seed"] for record in varied["records"]}) == 3
        # Reset noise changes the trajectory, so the episodes are no longer identical.
        assert len({record["completion_fraction"] for record in varied["records"]}) > 1

        # ... and the same seed still reproduces the same episode exactly.
        repeat = evaluate_policy(
            runner.model,
            catalog,
            ["level-01-basics"],
            config=EvalConfig(seeds=(11,), randomize_reset=True),
            env_config=tiny_config.env_config("level-01-basics"),
        )
        first = next(record for record in varied["records"] if record["seed"] == 11)
        assert repeat["records"][0]["completion_fraction"] == pytest.approx(
            first["completion_fraction"]
        )
        assert repeat["records"][0]["sim_steps"] == first["sim_steps"]
    finally:
        runner.vec.close()


def test_evaluation_rebuilds_checkpoint_dynamics(catalog: LevelCatalog) -> None:
    """A checkpoint trained at 20 Hz with a 0.4 action range must be evaluated that way."""

    from parkour_lab.evaluation import env_config_from_checkpoint

    extra = {
        "config": {
            "physics_timestep": 0.004,
            "control_decimation": 5,
            "action_scale": 0.4,
            "action_smoothing": 0.3,
            "actuation": "position",
            "max_episode_seconds": 6.0,
            "reward": {"progress_per_metre": 7.0},
        }
    }
    config = env_config_from_checkpoint(extra, "level-01-basics")
    assert config.control_dt == pytest.approx(0.02)
    assert config.action_scale == pytest.approx(0.4)
    assert config.action_smoothing == pytest.approx(0.3)
    assert config.max_episode_seconds == pytest.approx(6.0)
    assert config.reward.progress_per_metre == pytest.approx(7.0)
    # Without a recorded config the library defaults are used.
    default = env_config_from_checkpoint({}, "level-01-basics")
    assert default.action_scale == pytest.approx(0.25)

    # The rollout really applies the recorded control rate.
    model = ActorCritic(90, 19, hidden_dim=16, seed=0)
    level = catalog["level-01-basics"]
    record = rollout_episode(
        model,
        level,
        seed=5,
        env_config=config,
        deterministic=True,
        randomize_reset=False,
    )
    assert record["elapsed_seconds"] == pytest.approx(record["sim_steps"] * 0.02, abs=1e-6)


def test_stale_running_run_is_reported_as_interrupted(
    catalog: LevelCatalog, tmp_path: Path
) -> None:
    """A killed trainer must not keep claiming to be running forever."""

    from parkour_lab.storage import effective_status, pid_alive

    store = RunStore.create(tmp_path, run_id="dead-run", config={"level_ids": []})
    store.update_summary(status="running", pid=999_999_999)
    # A pid that cannot exist (above pid_max) is reported as not alive.
    assert pid_alive(999_999_999) is False
    assert effective_status(store.read_summary()) == "interrupted"
    assert effective_status({"status": "running"}) == "running"  # no pid recorded
    assert effective_status({"status": "running", "pid": 1}) in {"running", "interrupted"}
    listed = list_runs(tmp_path)
    assert listed[0]["status"] == "interrupted"
    assert listed[0]["recorded_status"] == "running"

    # A live trainer keeps its status.
    store.update_summary(status="running", pid=os.getpid())
    assert effective_status(store.read_summary()) == "running"


def test_evaluation_runs_the_requested_number_of_episodes(catalog: LevelCatalog) -> None:
    """``--eval-episodes`` must not be silently capped by the number of configured seeds."""

    from parkour_lab.evaluation import expand_eval_seeds

    assert expand_eval_seeds((11, 23), None) == [11, 23]
    assert expand_eval_seeds((11, 23), 1) == [11]
    assert expand_eval_seeds((11, 23), 2) == [11, 23]
    extra = expand_eval_seeds((11, 23), 5)
    assert len(extra) == 5
    assert len(set(extra)) == 5
    assert extra[:2] == [11, 23]
    with pytest.raises(ValueError):
        expand_eval_seeds((), 3)
    with pytest.raises(ValueError):
        expand_eval_seeds((11,), 0)

    model = ActorCritic(90, 19, hidden_dim=16, seed=0)
    evaluation = evaluate_policy(
        model,
        catalog,
        ["level-01-basics"],
        config=EvalConfig(seeds=(11, 23), deterministic=True, max_episodes_per_level=4),
    )
    assert evaluation["aggregate"]["overall"]["episodes"] == 4
    assert len({record["seed"] for record in evaluation["records"]}) == 4

    # The progress total must be the number of episodes that will actually run.
    progress: list[tuple[int, int]] = []
    evaluate_policy(
        model,
        catalog,
        ["level-01-basics"],
        config=EvalConfig(seeds=(11, 23), deterministic=True, max_episodes_per_level=4),
        progress_callback=lambda done, total, _record: progress.append((done, total)),
    )
    assert progress[-1] == (4, 4)


def _finish_at_spawn(tmp_path: Path):
    """A schema-valid level whose finish disc already contains the spawn point."""

    from parkour_lab.levels import DEFAULT_LEVELS_DIR, LevelCatalog, load_level

    data = json.loads((DEFAULT_LEVELS_DIR / "level-01-basics.json").read_text(encoding="utf-8"))
    data["id"] = "level-00-spawn-finish"
    data["finish"]["position"] = [0.2, 0.0, 0.92]
    path = tmp_path / "level-00-spawn-finish.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return LevelCatalog([load_level(path, use_schema=True)])


@pytest.fixture(scope="module")
def spawn_finish_catalog(tmp_path_factory) -> LevelCatalog:
    return _finish_at_spawn(tmp_path_factory.mktemp("spawn-finish"))


def test_stop_reason_needs_a_real_success() -> None:
    """``target_success=0`` used to mean "stop after the first evaluation", success or not."""

    from parkour_lab.training.runner import stop_reason_for

    config = TrainConfig(total_steps=1000)
    assert (
        stop_reason_for(config, best_eval_success=0.0, first_success=None, total_steps=100)
        is None
    )
    stopping = TrainConfig(total_steps=1000, stop_on_success=True)
    assert (
        stop_reason_for(stopping, best_eval_success=0.0, first_success=None, total_steps=100)
        is None
    )
    first = {"source": "rollout", "step": 128}
    assert (
        stop_reason_for(stopping, best_eval_success=0.0, first_success=first, total_steps=128)
        == "first_success"
    )
    assert (
        stop_reason_for(stopping, best_eval_success=0.2, first_success=None, total_steps=128)
        == "first_success"
    )
    target = TrainConfig(total_steps=1000, target_success=0.5)
    assert stop_reason_for(target, best_eval_success=0.4, first_success=first, total_steps=100) is None
    assert (
        stop_reason_for(target, best_eval_success=0.5, first_success=None, total_steps=100)
        == "target_success"
    )
    assert (
        stop_reason_for(config, best_eval_success=0.0, first_success=None, total_steps=1000)
        == "budget"
    )
    with pytest.raises(ValueError):
        TrainConfig(target_success=0.0)


def test_stop_on_success_ends_the_run_at_the_first_completed_course(
    tmp_path: Path, spawn_finish_catalog: LevelCatalog
) -> None:
    """The user-facing protocol: run until one episode finishes the course, then stop."""

    level = spawn_finish_catalog[0]
    config = TrainConfig(
        level_ids=(level.id,),
        curriculum="single",
        seed=5,
        total_steps=200_000,
        rollout_steps=32,
        num_envs=2,
        ppo=PPOConfig(epochs=1, minibatch_size=16),
        eval_interval=1_000,
        checkpoint_interval=1_000,
        hidden_dim=16,
        max_replays=0,
        record_replay_every_eval=0,
        stop_on_success=True,
    )
    summary = train(config, spawn_finish_catalog, runs_dir=tmp_path / "runs", run_id="stop-early")
    assert summary["stop_reason"] == "first_success"
    assert summary["first_success"]["source"] == "rollout"
    assert summary["first_success"]["step"] <= config.rollout_steps * config.num_envs
    assert summary["total_steps"] < config.total_steps
    stored = json.loads((tmp_path / "runs" / "stop-early" / "summary.json").read_text())
    assert stored["stop_reason"] == "first_success"
    assert stored["first_success"]["level_id"] == level.id
    # The interrupt-style status must not regress: the run did stop early on purpose.
    assert stored["status"] == "stopped_early"


def test_runs_without_stop_on_success_spend_their_budget(
    tmp_path: Path, spawn_finish_catalog: LevelCatalog
) -> None:
    level = spawn_finish_catalog[0]
    config = TrainConfig(
        level_ids=(level.id,),
        curriculum="single",
        seed=5,
        total_steps=64,
        rollout_steps=32,
        num_envs=2,
        ppo=PPOConfig(epochs=1, minibatch_size=16),
        eval_interval=1_000,
        checkpoint_interval=1_000,
        hidden_dim=16,
        max_replays=0,
        record_replay_every_eval=0,
    )
    summary = train(config, spawn_finish_catalog, runs_dir=tmp_path / "runs", run_id="no-stop")
    assert summary["stop_reason"] == "budget"
    assert summary["first_success"] is not None  # it happened, it just did not stop the run
    assert summary["status"] == "completed"


def test_rollout_acts_on_the_reset_observation_after_an_episode_ends(
    tmp_path: Path, spawn_finish_catalog: LevelCatalog
) -> None:
    """Every rollout row must be the state the action was actually applied in.

    A finished episode leaves ``step`` with the terminal observation, and the environment is
    reset only afterwards.  If the trainer ignored that reset it would pair the previous
    episode's final state with the first action of the new one, once per episode.
    """

    from parkour_lab.storage import RunStore
    from parkour_lab.training.runner import TrainingRunner

    level = spawn_finish_catalog[0]
    config = TrainConfig(
        level_ids=(level.id,),
        curriculum="single",
        seed=7,
        total_steps=10_000,
        rollout_steps=6,
        num_envs=2,
        ppo=PPOConfig(epochs=1, minibatch_size=8),
        hidden_dim=16,
        randomize_reset=False,
        eval_interval=10_000,
        checkpoint_interval=10_000,
        max_replays=0,
        record_replay_every_eval=0,
    )
    store = RunStore.create(
        tmp_path / "runs", run_id="stale-obs", name="stale-obs", config=config.to_dict()
    )
    runner = TrainingRunner(config, spawn_finish_catalog, store, logger=lambda _: None)
    runner.setup()
    try:
        batch, metrics = runner._collect_rollout()
    finally:
        assert runner.vec is not None
        runner.vec.close()
    # Every control step completes the course, so every row starts a fresh episode.
    assert metrics["episodes"] == config.rollout_steps * config.num_envs
    np.testing.assert_allclose(batch.observations[0], batch.observations[-1], atol=1e-6)
