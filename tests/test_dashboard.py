"""Tests for the FastAPI research dashboard.

The dashboard never simulates physics in the web process: it reads the same run
directories the trainer writes.  These tests therefore build a synthetic run on disk
(cheap and deterministic) and assert the JSON contracts the browser front-end relies on,
plus the validation and safety rules around starting a training job.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from parkour_lab.dashboard.server import create_app
from parkour_lab.levels import DEFAULT_LEVELS_DIR
from parkour_lab.storage import RunStore, utc_timestamp

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run(runs_dir: Path, run_id: str = "demo-01") -> RunStore:
    """Create a run directory with exactly the artefacts the trainer produces."""

    store = RunStore.create(
        runs_dir,
        run_id=run_id,
        config={
            "name": "demo",
            "notes": "",
            "level_ids": ["level-01-basics"],
            "curriculum": "single",
            "seed": 4,
            "total_steps": 4096,
            "rollout_steps": 128,
            "num_envs": 2,
            "workers": 1,
            "action_scale": 0.25,
            "action_smoothing": 0.6,
            "actuation": "position",
            "physics_timestep": 0.004,
            "control_decimation": 5,
            "initial_log_std": -1.0,
        },
    )
    store.write_json(
        "prov.json",
        {"created_utc": utc_timestamp()},
    )
    _write_csv(
        store.paths.training_csv,
        [
            {
                "iteration": index,
                "total_steps": index * 1024,
                "steps_per_second": 1900.0,
                "episode_return": 1.0 * index,
                "episode_length": 120.0,
                "success_rate": 0.0,
                "mean_completion": 0.1 * index / 4,
                "policy_loss": 0.01,
                "value_loss": 20.0 - index,
                "entropy": 7.5,
                "approx_kl": 0.01,
                "clip_fraction": 0.2,
                "explained_variance": 0.8,
                "eval_success_rate": "",
                "eval_mean_completion": "",
            }
            for index in range(1, 5)
        ],
    )
    _write_jsonl(
        store.paths.episodes_jsonl,
        [
            {
                "level_id": "level-01-basics",
                "iteration": 1 + index // 2,
                "total_steps": 1024 * (1 + index // 2),
                "seed": index,
                "success": index == 3,
                "termination_reason": "course_complete" if index == 3 else "collapsed",
                "completion_fraction": 1.0 if index == 3 else 0.12,
                "episode_return": 5.0,
                "duration_seconds": 3.0,
            }
            for index in range(4)
        ],
    )
    _write_csv(
        store.paths.evaluation_csv,
        [
            {
                "iteration": 2,
                "total_steps": 2048,
                "level_id": "level-01-basics",
                "episodes": 2,
                "success_rate": 0.5,
                "mean_completion": 0.4,
                "mean_return": 6.0,
            }
        ],
    )
    evaluation = {
        "iteration": 2,
        "total_steps": 2048,
        "levels": ["level-01-basics"],
        "aggregate": {
            "overall": {
                "episodes": 2,
                "successes": 1,
                "success_rate": 0.5,
                "mean_completion": 0.4,
                "std_completion": 0.0,
                "mean_return": 6.0,
                "mean_duration": 4.0,
            },
            "per_level": {
                "level-01-basics": {
                    "level_id": "level-01-basics",
                    "episodes": 2,
                    "successes": 1,
                    "success_rate": 0.5,
                    "mean_completion": 0.4,
                    "std_completion": 0.0,
                    "mean_return": 6.0,
                    "mean_duration": 4.0,
                }
            },
            "terminations": {"course_complete": 1, "collapsed": 1},
        },
        "records": [],
    }
    store.write_json("evaluation/iter_00002.json", evaluation)
    frames = [
        {
            "t": 0.0,
            "bodies": {"pelvis": [0.0, 0.0, 0.94, 1.0, 0.0, 0.0, 0.0]},
            "movers": {},
        },
        {
            "t": 0.02,
            "bodies": {"pelvis": [0.1, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0]},
            "movers": {},
        },
    ]
    payload = {
        "run_id": run_id,
        "level_id": "level-01-basics",
        "level_title": "First Steps",
        "seed": 11,
        "success": False,
        "termination_reason": "collapsed",
        "completion_fraction": 0.12,
        "episode_return": 5.0,
        "frames": frames,
        "level": {"id": "level-01-basics", "obstacles": []},
    }
    with gzip.open(
        store.paths.replay_dir / "iter00002-level-01-basics-seed11.json.gz", "wt", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle)
    store.update_summary(
        status="completed",
        total_steps=4096,
        iterations=4,
        episodes=4,
        best_eval_success=0.5,
        best_eval_completion=0.4,
        best_eval_iteration=2,
    )
    return store


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir)
    app = create_app(runs_dir=runs_dir, levels_dir=DEFAULT_LEVELS_DIR)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_versions_and_directories(client: TestClient, tmp_path: Path) -> None:
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["runs_dir"] == str((tmp_path / "runs").resolve())
    assert payload["mujoco"].count(".") >= 1
    assert payload["active_job"] is None


def test_levels_endpoint_lists_every_manifest(client: TestClient) -> None:
    levels = client.get("/api/levels").json()["levels"]
    assert len(levels) == 22
    first = levels[0]
    assert first["id"] == "level-01-basics"
    for key in ("title", "difficulty_label", "spawn_support_z", "obstacles", "finish"):
        assert key in first


def test_run_list_and_detail_contract(client: TestClient) -> None:
    runs = client.get("/api/runs").json()["runs"]
    assert [run["run_id"] for run in runs] == ["demo-01"]
    summary = runs[0]
    assert summary["level_ids"] == ["level-01-basics"]
    assert summary["total_steps"] == 4096
    assert summary["best_eval_success"] == pytest.approx(0.5)
    assert summary["num_envs"] == 2
    assert summary["checkpoints"] == 0

    detail = client.get("/api/runs/demo-01").json()
    for key in (
        "config",
        "summary",
        "provenance",
        "training",
        "episode_aggregate",
        "evaluation_history",
        "latest_evaluation",
        "checkpoints",
        "replays",
    ):
        assert key in detail
    training = detail["training"]
    assert [row["total_steps"] for row in training] == [1024, 2048, 3072, 4096]
    assert training[0]["eval_success_rate"] is None

    aggregate = detail["episode_aggregate"]
    assert aggregate["episodes"] == 4
    per_level = aggregate["levels"][0]
    assert per_level["level_id"] == "level-01-basics"
    assert per_level["success_rate"] == pytest.approx(0.25)
    assert aggregate["terminations"]["collapsed"] == 3
    assert aggregate["timeline"]

    history = detail["evaluation_history"]
    assert history and history[0]["success_rate"] == pytest.approx(0.5)
    assert detail["latest_evaluation"]["aggregate"]["overall"]["successes"] == 1
    assert [replay["name"] for replay in detail["replays"]] == [
        "iter00002-level-01-basics-seed11.json.gz"
    ]


def test_episodes_endpoint_paginates_and_filters(client: TestClient) -> None:
    payload = client.get("/api/runs/demo-01/episodes", params={"limit": 2}).json()
    assert payload["total"] == 4
    assert len(payload["episodes"]) == 2
    filtered = client.get(
        "/api/runs/demo-01/episodes", params={"level": "level-99-missing"}
    ).json()
    assert filtered["total"] == 0


def test_replay_endpoint_returns_frames_and_rejects_traversal(client: TestClient) -> None:
    payload = client.get("/api/runs/demo-01/replay/iter00002-level-01-basics-seed11.json.gz")
    assert payload.status_code == 200
    body = payload.json()
    assert body["level_id"] == "level-01-basics"
    assert len(body["frames"]) == 2
    assert body["frames"][1]["bodies"]["pelvis"][0] == pytest.approx(0.1)

    traversal = client.get("/api/runs/demo-01/replay/..%2Fconfig.json")
    assert traversal.status_code in (400, 404)
    wrong_suffix = client.get("/api/runs/demo-01/replay/summary.json")
    assert wrong_suffix.status_code == 400
    missing = client.get("/api/runs/demo-01/replay/nope.json.gz")
    assert missing.status_code == 404


def test_unknown_run_and_bad_identifiers(client: TestClient) -> None:
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.get("/api/runs/bad%20id").status_code in (400, 404)
    assert client.get("/api/compare", params={"a": "demo-01", "b": "nope"}).status_code == 404


def test_compare_two_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "demo-01")
    _make_run(runs_dir, "demo-02")
    app = create_app(runs_dir=runs_dir, levels_dir=DEFAULT_LEVELS_DIR)
    with TestClient(app) as test_client:
        payload = test_client.get(
            "/api/compare", params={"a": "demo-01", "b": "demo-02"}
        ).json()
        assert payload["rows"]
        row = payload["rows"][0]
        assert row["level_id"] == "level-01-basics"
        assert row["delta_success"] == pytest.approx(0.0)


def test_training_request_validation(client: TestClient) -> None:
    unknown = client.post("/api/train", json={"levels": ["level-99-missing"], "steps": 4096})
    assert unknown.status_code == 400
    assert "unknown levels" in unknown.json()["detail"]

    too_big = client.post("/api/train", json={"levels": ["level-01-basics"], "steps": 10**9})
    assert too_big.status_code == 422

    single_multi = client.post(
        "/api/train",
        json={
            "levels": ["level-01-basics", "level-02-step-up"],
            "curriculum": "single",
            "steps": 4096,
        },
    )
    assert single_multi.status_code == 400


def test_training_request_builds_validated_cli_arguments(client: TestClient) -> None:
    response = client.post(
        "/api/train",
        json={
            "levels": ["level-01-basics"],
            "steps": 2000,
            "envs": 2,
            "seed": 5,
            "name": "settings-test",
            "settings": {
                "physics_timestep": 0.004,
                "control_decimation": 5,
                "action_scale": 0.3,
                "stagnation_seconds": 2.0,
                "reward_ramp_fraction": 0.1,
            },
            "reward": {"gait_per_second": 6.0, "progress_per_metre": 6.0},
        },
    )
    assert response.status_code == 200
    command = response.json()["command"]
    pairs = dict(zip(command, command[1:]))
    assert pairs["--physics-timestep"] == "0.004"
    assert pairs["--decimation"] == "5"
    assert pairs["--action-scale"] == "0.3"
    assert pairs["--stagnation-seconds"] == "2"
    assert pairs["--reward-ramp-fraction"] == "0.1"
    reward_sets = [command[i + 1] for i, item in enumerate(command) if item == "--reward-set"]
    assert "gait_per_second=6" in reward_sets
    assert "progress_per_metre=6" in reward_sets
    client.post(f"/api/jobs/{response.json()['job_id']}/stop")


def test_training_request_rejects_unknown_or_out_of_range_settings(client: TestClient) -> None:
    unknown = client.post(
        "/api/train",
        json={"levels": ["level-01-basics"], "steps": 2000, "reward": {"magic": 1.0}},
    )
    assert unknown.status_code == 400
    assert "unknown reward setting" in unknown.json()["detail"]

    forbidden = client.post(
        "/api/train",
        json={"levels": ["level-01-basics"], "steps": 2000, "settings": {"gravity": -9.81}},
    )
    assert forbidden.status_code == 400
    assert "unknown training setting" in forbidden.json()["detail"]

    too_wide = client.post(
        "/api/train",
        json={"levels": ["level-01-basics"], "steps": 2000, "settings": {"action_scale": 9.0}},
    )
    assert too_wide.status_code == 400
    assert "must be between" in too_wide.json()["detail"]


def test_training_job_lifecycle(client: TestClient) -> None:
    """Start a very short real training job through the API, then stop it."""

    response = client.post(
        "/api/train",
        json={
            "levels": ["level-01-basics"],
            "steps": 1000,
            "envs": 1,
            "seed": 2,
            "name": "api-test",
            "notes": "started by tests/test_dashboard.py",
        },
    )
    assert response.status_code == 200
    job = response.json()
    assert job["run_id"].endswith("-api-test-" + job["run_id"].rsplit("-", 1)[-1])
    assert job["status"] == "running"
    assert "train" in job["command"]
    assert job["command"][job["command"].index("--steps") + 1] == "1000"

    health = client.get("/api/health").json()
    assert health["active_job"]["job_id"] == job["job_id"]

    second = client.post("/api/train", json={"levels": ["level-01-basics"], "steps": 1000})
    assert second.status_code == 409

    log = client.get(f"/api/jobs/{job['job_id']}/log").json()
    assert log["job"]["job_id"] == job["job_id"]
    assert isinstance(log["lines"], list)

    stopped = client.post(f"/api/jobs/{job['job_id']}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] in ("stopped", "finished", "failed")
    assert client.get("/api/jobs/nope/log").status_code == 404


def test_static_frontend_is_served(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "<title>" in index.text

    app_js = client.get("/static/app.js")
    assert app_js.status_code == 200
    assert "javascript" in app_js.headers["content-type"]

    module = client.get("/static/vendor/three.module.js")
    assert module.status_code == 200
    assert "three" in module.text[:2000].lower() or len(module.text) > 100_000

    for asset in ("styles.css", "favicon.svg"):
        assert client.get(f"/static/{asset}").status_code == 200


def test_benchmark_endpoint_is_bounded(client: TestClient) -> None:
    payload = client.get("/api/benchmark", params={"envs": "1,2", "steps": 20}).json()
    results = payload["results"]
    assert len(results) == 2
    assert all(entry["steps_per_second"] > 0 for entry in results)
    assert [entry["num_envs"] for entry in results] == [1, 2]


def test_missing_runs_directory_is_created(tmp_path: Path) -> None:
    runs_dir = tmp_path / "fresh-runs"
    app = create_app(runs_dir=runs_dir, levels_dir=DEFAULT_LEVELS_DIR)
    with TestClient(app) as test_client:
        assert test_client.get("/api/runs").json() == {"runs": []}
    assert runs_dir.is_dir()


def test_run_list_separates_progress_from_the_step_budget(tmp_path: Path) -> None:
    """A live run must not look finished just because the summary holds its target."""

    runs_dir = tmp_path / "runs"
    store = _make_run(runs_dir, run_id="live-01")
    store.write_json("config.json", {**store.read_config(), "total_steps": 1_000_000})
    # What the trainer writes at startup: the budget, plus its own pid.
    store.update_summary(status="running", total_steps=1_000_000, pid=os.getpid())
    app = create_app(runs_dir=runs_dir, levels_dir=DEFAULT_LEVELS_DIR)
    with TestClient(app) as test_client:
        run = test_client.get("/api/runs").json()["runs"][0]
    # Four training rows at 1024..4096 steps, budget of 1M: progress is what is real.
    assert run["progress_steps"] == 4096
    assert run["target_steps"] == 1_000_000
    assert run["total_steps"] == 1_000_000


def test_running_run_without_pid_goes_stale(tmp_path: Path) -> None:
    """Runs recorded before pid tracking existed must still be able to look interrupted."""

    from parkour_lab.storage import effective_status, last_activity, list_runs

    runs_dir = tmp_path / "runs"
    store = _make_run(runs_dir, run_id="old-01")
    store.update_summary(status="running")
    summary = store.read_summary()
    assert "pid" not in summary
    # Freshly written metrics: still plausible as a live run.
    assert effective_status(summary, last_activity=last_activity(store.paths.root)) == "running"
    # Two hours without touching a metrics file: reported as interrupted.
    stale = last_activity(store.paths.root) - 2 * 3600
    assert effective_status(summary, last_activity=stale) == "interrupted"
    oldest = last_activity(store.paths.root) - 2 * 3600
    for path in store.paths.metrics_dir.glob("*"):
        os.utime(path, (oldest, oldest))
    os.utime(store.paths.summary, (oldest, oldest))
    listed = {run["run_id"]: run for run in list_runs(runs_dir)}
    assert listed["old-01"]["status"] == "interrupted"
    assert listed["old-01"]["recorded_status"] == "running"


def test_best_evaluation_breaks_success_ties_on_completion(tmp_path: Path) -> None:
    """With success still at zero, "best" must mean furthest progress, not the first record."""

    from parkour_lab.storage import list_runs

    runs_dir = tmp_path / "runs"
    store = _make_run(runs_dir, run_id="ties-01")
    for path in store.paths.evaluation_dir.glob("*.json"):
        path.unlink()
    for iteration, (success, completion) in enumerate([(0.0, 0.10), (0.0, 0.42), (0.0, 0.25)]):
        store.write_json(
            f"evaluation/{iteration:05d}.json",
            {
                "iteration": iteration,
                "total_steps": 1024 * (iteration + 1),
                "aggregate": {
                    "overall": {
                        "episodes": 2,
                        "success_rate": success,
                        "mean_completion": completion,
                    }
                },
            },
        )
    run = [entry for entry in list_runs(runs_dir) if entry["run_id"] == "ties-01"][0]
    assert run["best_eval_success"] == pytest.approx(0.0)
    assert run["best_eval_completion"] == pytest.approx(0.42)
