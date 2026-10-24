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
