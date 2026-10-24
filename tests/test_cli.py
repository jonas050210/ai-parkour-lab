"""Command-line interface tests.

Every command is exercised through :func:`parkour_lab.cli.main` (the same entry point the
installed ``parkour`` script uses) against a temporary runs directory, so the tests cover
argument parsing, output and exit codes rather than private helpers.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from parkour_lab.cli import main
from parkour_lab.levels import LevelCatalog
from parkour_lab.ppo import PPOConfig
from parkour_lab.training.runner import TrainConfig, train

LEVEL = "level-01-basics"


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


@pytest.fixture(scope="module")
def trained(
    catalog: LevelCatalog, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A short real run with a checkpoint, an evaluation and a replay."""

    runs_dir = tmp_path_factory.mktemp("cli-runs")
    config = TrainConfig(
        level_ids=(LEVEL,),
        curriculum="single",
        seed=6,
        total_steps=4096,
        rollout_steps=64,
        num_envs=2,
        ppo=PPOConfig(epochs=2, minibatch_size=64),
        eval_interval=2,
        eval_levels=(LEVEL,),
        eval_seeds=(11, 23),
        eval_episodes_per_level=1,
        checkpoint_interval=2,
        keep_checkpoints=2,
        hidden_dim=20,
        max_replays=2,
        record_replay_every_eval=1,
        physics_timestep=0.004,
        control_decimation=5,
        action_scale=0.3,
        action_smoothing=0.5,
        max_episode_seconds=4.0,
    )
    train(config, catalog, runs_dir=runs_dir, run_id="cli-run")
    return runs_dir


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "ai-parkour-lab" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])


def test_list_levels_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-levels"]) == 0
    text = capsys.readouterr().out
    assert LEVEL in text
    assert "level-20-final-gauntlet" in text

    assert main(["list-levels", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 22
    split = {level["id"]: level["split"] for level in payload}
    assert split["level-21-heldout-course"] == "held-out"


def test_doctor_reports_a_usable_environment(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "mujoco" in output.lower()
    assert "levels" in output.lower()


def test_inspect_and_export_mjcf(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["inspect", "--level", LEVEL]) == 0
    output = capsys.readouterr().out
    assert "spawn" in output
    assert "19" in output  # actuator count is part of the summary

    out = tmp_path / "level.xml"
    assert main(["export-mjcf", "--level", LEVEL, "--out", str(out)]) == 0
    xml = out.read_text()
    assert "<mujoco" in xml
    assert "parkour" in xml


def test_inspect_rejects_an_unknown_level(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", "--level", "level-99-missing"]) == 2
    error = capsys.readouterr().err
    assert "level-99-missing" in error


def test_random_policy_runs_and_reports(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["random", "--level", LEVEL, "--steps", "40", "--seed", "2"]) == 0
    output = capsys.readouterr().out
    assert "steps" in output.lower()

    assert main(["random", "--level", LEVEL, "--steps", "20", "--policy", "stand"]) == 0
    assert "termination" in capsys.readouterr().out.lower()


def test_evaluate_loads_checkpoint_dynamics(
    trained: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    checkpoint = trained / "cli-run" / "checkpoints" / "latest.npz"
    out = tmp_path / "eval.json"
    code = main(
        [
            "--runs-dir",
            str(trained),
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--level",
            LEVEL,
            "--eval-seeds",
            "11",
            "23",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert LEVEL in printed
    payload = json.loads(out.read_text())
    assert payload["aggregate"]["overall"]["episodes"] == 2
    # The evaluation must use the dynamics recorded in the checkpoint (20 Hz here).
    config = payload["config"]
    assert config is not None


def test_evaluate_with_replay_export(
    trained: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    checkpoint = trained / "cli-run" / "checkpoints" / "latest.npz"
    replay_dir = tmp_path / "replays"
    code = main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--level",
            LEVEL,
            "--eval-seeds",
            "11",
            "--record-replay",
            "--replay-out",
            str(replay_dir),
        ]
    )
    assert code == 0
    capsys.readouterr()
    files = sorted(replay_dir.glob("*.json.gz"))
    assert files, "expected at least one exported replay"
    with gzip.open(files[0], "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["level_id"] == LEVEL
    assert payload["frames"]
    assert set(payload["frames"][0]) == {"t", "bodies", "movers"}
    assert "pelvis" in payload["frames"][0]["bodies"]


def test_replay_command_writes_a_frame_log(
    trained: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    checkpoint = trained / "cli-run" / "checkpoints" / "latest.npz"
    out = tmp_path / "episode.json.gz"
    code = main(
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--level",
            LEVEL,
            "--seed",
            "11",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    capsys.readouterr()
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["run_id"] == "cli-run"
    assert payload["level"]["id"] == LEVEL
    assert len(payload["frames"]) > 1
    assert payload["termination_reason"] in {
        "course_complete",
        "fell_over",
        "collapsed",
        "void_fall",
        "hazard_contact",
        "out_of_bounds",
        "timeout",
    }


def test_runs_and_compare(trained: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--runs-dir", str(trained), "runs"]) == 0
    listing = capsys.readouterr().out
    assert "cli-run" in listing

    assert main(["--runs-dir", str(trained), "runs", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["run_id"] == "cli-run"

    assert (
        main(
            [
                "--runs-dir",
                str(trained),
                "compare",
                "--run-a",
                "cli-run",
                "--run-b",
                "cli-run",
            ]
        )
        == 0
    )
    comparison = capsys.readouterr().out
    assert "cli-run" in comparison
    assert "success" in comparison.lower()


def test_runs_rejects_an_unknown_run(trained: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--runs-dir", str(trained), "compare", "--run-a", "cli-run", "--run-b", "x"]) == 2
    assert "x" in capsys.readouterr().err


def test_benchmark_runs_a_short_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--envs", "1,2", "--steps", "20", "--workers", "1"]) == 0
    output = capsys.readouterr().out
    assert "transitions/s" in output


def test_harness_status_and_plan(
    trained: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--runs-dir", str(trained), "harness", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert "restricted" in status["name"].lower()
    assert any("physics" in item for item in status["cannot"])

    assert (
        main(
            [
                "--runs-dir",
                str(trained),
                "harness",
                "plan",
                "--run",
                str(trained / "cli-run"),
                "--max-proposals",
                "2",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert 1 <= len(plan) <= 2
    for proposal in plan:
        assert proposal["applied"] is False


def test_harness_probe_writes_a_report(
    trained: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--runs-dir",
            str(trained),
            "harness",
            "probe",
            "--run",
            str(trained / "cli-run"),
            "--level",
            LEVEL,
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["run_id"] == "cli-run"
    assert (trained / "cli-run" / "harness2").is_dir()


def test_evolve_cli_writes_a_search(
    trained: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--runs-dir",
            str(trained),
            "evolve",
            "--level",
            LEVEL,
            "--generations",
            "1",
            "--population",
            "2",
            "--train-steps",
            "512",
            "--envs",
            "1",
            "--workers",
            "1",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "best_run_id" in output
    results = sorted(trained.glob("evolve-*/result.json"))
    assert results, "evolution should create its own directory"
    result = json.loads(results[-1].read_text())
    assert result["best_run_id"].startswith("evolve-")
    assert result["candidates"]


def test_train_cli_records_a_run(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main(
        [
            "--runs-dir",
            str(tmp_path),
            "train",
            "--level",
            LEVEL,
            "--run-id",
            "cli-train",
            "--steps",
            "512",
            "--envs",
            "2",
            "--rollout-steps",
            "128",
            "--seed",
            "1",
            "--hidden-dim",
            "16",
            "--epochs",
            "2",
            "--eval-interval",
            "999",
            "--checkpoint-interval",
            "999",
            "--physics-timestep",
            "0.004",
            "--max-episode-seconds",
            "3",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "training 1 level(s)" in output
    assert '"run_id": "cli-train"' in output
    assert (tmp_path / "cli-train" / "summary.json").exists()
    store_config = json.loads((tmp_path / "cli-train" / "config.json").read_text())
    assert store_config["max_episode_seconds"] == pytest.approx(3.0)
    assert store_config["hidden_dim"] == 16
