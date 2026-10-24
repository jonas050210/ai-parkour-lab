"""FastAPI application for the local research dashboard.

The API is read-only for saved runs plus one explicitly bounded control surface: starting
and stopping a local training process with validated parameters.  It never simulates
physics inside the web process, never edits a checkpoint, and never promotes a model.
Everything it displays comes from the run directories written by the trainer.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..levels import DEFAULT_LEVELS_DIR, LevelCatalog, LevelError
from ..storage import DEFAULT_RUNS_DIR, RunStore, list_runs, read_json

STATIC_DIR = Path(__file__).parent / "static"
MAX_STEPS = 5_000_000
MAX_ENVS = 16
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


# --------------------------------------------------------------------------------------
# Run data helpers
# --------------------------------------------------------------------------------------
class _Cache:
    """Tiny mtime-keyed JSON/CSV cache so the dashboard stays responsive."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, path: Path, loader: Any) -> Any:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return loader() if path.suffix else None
        key = str(path)
        cached = self._entries.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        value = loader()
        self._entries[key] = (mtime, value)
        return value


_CACHE = _Cache()


def _store(runs_dir: Path, run_id: str) -> RunStore:
    if not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    root = (runs_dir / run_id).resolve()
    if runs_dir.resolve() not in root.parents:
        raise HTTPException(status_code=400, detail="invalid run path")
    if not (root / "config.json").exists():
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return RunStore.load(root)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _training_rows(store: RunStore) -> list[dict[str, Any]]:
    rows = _CACHE.get(store.paths.training_csv, store.training_rows)
    cleaned = []
    for row in rows:
        cleaned.append(
            {
                "iteration": int(_float(row.get("iteration"))),
                "total_steps": int(_float(row.get("total_steps"))),
                "steps_per_second": _float(row.get("steps_per_second")),
                "episode_return": _float(row.get("episode_return")),
                "episode_length": _float(row.get("episode_length")),
                "success_rate": _float(row.get("success_rate")),
                "mean_completion": _float(row.get("mean_completion")),
                "policy_loss": _float(row.get("policy_loss")),
                "value_loss": _float(row.get("value_loss")),
                "entropy": _float(row.get("entropy")),
                "approx_kl": _float(row.get("approx_kl")),
                "explained_variance": _float(row.get("explained_variance")),
                "eval_success_rate": (
                    None
                    if row.get("eval_success_rate") in ("", None)
                    else _float(row.get("eval_success_rate"))
                ),
                "eval_mean_completion": (
                    None
                    if row.get("eval_mean_completion") in ("", None)
                    else _float(row.get("eval_mean_completion"))
                ),
            }
        )
    return cleaned


def _episode_aggregate(store: RunStore) -> dict[str, Any]:
    """Aggregate the episode JSONL into per-level and per-termination summaries."""

    episodes = _CACHE.get(store.paths.episodes_jsonl, lambda: list(store.episodes()))
    per_level: dict[str, dict[str, Any]] = {}
    terminations: dict[str, int] = {}
    timeline: list[dict[str, Any]] = []
    bucket: dict[int, dict[str, Any]] = {}
    for record in episodes:
        level_id = str(record.get("level_id", "unknown"))
        bucket_level = per_level.setdefault(
            level_id,
            {
                "level_id": level_id,
                "episodes": 0,
                "successes": 0,
                "completions": [],
                "returns": [],
                "durations": [],
                "terminations": {},
            },
        )
        bucket_level["episodes"] += 1
        bucket_level["successes"] += int(bool(record.get("success", False)))
        bucket_level["completions"].append(_float(record.get("completion_fraction")))
        bucket_level["returns"].append(_float(record.get("episode_return")))
        bucket_level["durations"].append(_float(record.get("elapsed_seconds")))
        reason = str(record.get("termination_reason", "unknown"))
        bucket_level["terminations"][reason] = bucket_level["terminations"].get(reason, 0) + 1
        terminations[reason] = terminations.get(reason, 0) + 1
        iteration = int(_float(record.get("iteration"), 0))
        chunk = bucket.setdefault(
            iteration // 10 * 10,
            {"iteration": iteration // 10 * 10, "episodes": 0, "successes": 0, "completion": 0.0},
        )
        chunk["episodes"] += 1
        chunk["successes"] += int(bool(record.get("success", False)))
        chunk["completion"] += _float(record.get("completion_fraction"))
    levels = []
    for data in sorted(per_level.values(), key=lambda item: item["level_id"]):
        episodes_count = max(1, data["episodes"])
        levels.append(
            {
                "level_id": data["level_id"],
                "episodes": data["episodes"],
                "success_rate": data["successes"] / episodes_count,
                "mean_completion": sum(data["completions"]) / episodes_count,
                "mean_return": sum(data["returns"]) / episodes_count,
                "mean_duration": sum(data["durations"]) / episodes_count,
                "terminations": data["terminations"],
            }
        )
    timeline = [
        {
            "iteration": chunk["iteration"],
            "episodes": chunk["episodes"],
            "success_rate": chunk["successes"] / max(1, chunk["episodes"]),
            "mean_completion": chunk["completion"] / max(1, chunk["episodes"]),
        }
        for chunk in sorted(bucket.values(), key=lambda item: item["iteration"])
    ]
    return {
        "episodes": len(episodes),
        "levels": levels,
        "terminations": dict(sorted(terminations.items(), key=lambda item: -item[1])),
        "timeline": timeline[-200:],
    }


def _evaluation_history(store: RunStore) -> list[dict[str, Any]]:
    rows = _CACHE.get(store.paths.evaluation_csv, store.evaluation_rows)
    return [
        {
            "iteration": int(_float(row.get("iteration"))),
            "total_steps": int(_float(row.get("total_steps"))),
            "episodes": int(_float(row.get("episodes"))),
            "success_rate": _float(row.get("success_rate")),
            "mean_completion": _float(row.get("mean_completion")),
            "mean_return": _float(row.get("mean_return")),
            "levels": row.get("levels", ""),
        }
        for row in rows
    ]


def _latest_evaluation(store: RunStore) -> dict[str, Any] | None:
    evaluations = _CACHE.get(
        store.paths.evaluation_dir,
        lambda: sorted(
            (
                record
                for record in (read_json(path) for path in store.paths.evaluation_dir.glob("*.json"))
                if isinstance(record, dict)
            ),
            key=lambda record: int(record.get("iteration", 0)),
        ),
    )
    if not evaluations:
        return None
    latest = evaluations[-1]
    return {
        "iteration": latest.get("iteration"),
        "total_steps": latest.get("total_steps"),
        "levels": latest.get("levels"),
        "aggregate": latest.get("aggregate"),
        "records": [
            {
                key: record.get(key)
                for key in (
                    "level_id",
                    "seed",
                    "success",
                    "completion_fraction",
                    "episode_return",
                    "termination_reason",
                    "elapsed_seconds",
                )
            }
            for record in latest.get("records", [])
        ],
    }


def _replay_payload(path: Path) -> dict[str, Any]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------------------
# Training jobs
# --------------------------------------------------------------------------------------
@dataclass
class TrainingJob:
    job_id: str
    run_id: str
    command: list[str]
    status: str = "starting"
    started_utc: float = field(default_factory=time.time)
    exit_code: int | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=4000))
    process: subprocess.Popen | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status,
            "started_utc": self.started_utc,
            "exit_code": self.exit_code,
            "command": self.command,
            "log_lines": len(self.log),
        }


#: Training knobs the dashboard may set, with the bounds the CLI accepts.
#: Keys are ``TrainConfig``/``RewardConfig`` field names so the mapping stays explicit.
PHYSICS_BOUNDS = {
    "physics_timestep": (0.001, 0.01),
    "control_decimation": (1, 20),
    "action_scale": (0.05, 0.8),
    "action_smoothing": (0.0, 0.95),
    "max_episode_seconds": (0.5, 60.0),
    "stagnation_seconds": (0.0, 30.0),
    "reward_ramp_fraction": (0.0, 0.9),
    "reward_scale": (0.01, 5.0),
    "learning_rate": (1e-5, 5e-3),
    "entropy_coefficient": (0.0, 0.05),
    "initial_log_std": (-4.0, 0.5),
}

REWARD_BOUNDS = {
    "progress_per_metre": (-20.0, 40.0),
    "forward_velocity_per_second": (-20.0, 40.0),
    "forward_velocity_target": (0.1, 5.0),
    "gait_per_second": (0.0, 40.0),
    "alive_per_second": (-5.0, 20.0),
    "upright_per_second": (-5.0, 20.0),
    "height_per_second": (-5.0, 20.0),
    "action_rate_penalty": (-1.0, 1.0),
    "torque_penalty": (-1e-3, 1e-3),
    "joint_velocity_penalty": (-1e-2, 1e-2),
    "lateral_drift_penalty": (-1.0, 1.0),
    "success_bonus": (0.0, 500.0),
    "speed_bonus": (0.0, 100.0),
    "fall_penalty": (0.0, 200.0),
    "hazard_penalty": (0.0, 200.0),
    "out_of_bounds_penalty": (0.0, 200.0),
}


class TrainRequest(BaseModel):
    levels: list[str] = Field(default_factory=list)
    steps: int = Field(default=200_000, ge=1_000, le=MAX_STEPS)
    envs: int = Field(default=4, ge=1, le=MAX_ENVS)
    workers: int = Field(default=1, ge=1, le=MAX_ENVS)
    curriculum: str = Field(default="adaptive", pattern="^(adaptive|uniform|single)$")
    seed: int = Field(default=0, ge=0, le=1_000_000)
    name: str = Field(default="", max_length=48)
    notes: str = Field(default="", max_length=500)
    max_minutes: float | None = Field(default=None, ge=0.1, le=1440)
    rollout_steps: int = Field(default=512, ge=32, le=4096)
    hidden_dim: int = Field(default=128, ge=8, le=1024)
    eval_interval: int = Field(default=20, ge=1, le=10_000)
    checkpoint_interval: int = Field(default=20, ge=1, le=10_000)
    #: Optional physics/PPO overrides; keys must exist in ``PHYSICS_BOUNDS``.
    settings: dict[str, float] = Field(default_factory=dict)
    #: Optional reward overrides; keys must exist in ``REWARD_BOUNDS``.
    reward: dict[str, float] = Field(default_factory=dict)


def _bounded(
    overrides: dict[str, float], bounds: dict[str, tuple[float, float]], kind: str
) -> list[tuple[str, str]]:
    """Validate override names/ranges and return ``(name, cli_text)`` pairs."""

    cleaned: list[tuple[str, str]] = []
    for name, value in overrides.items():
        if name not in bounds:
            raise HTTPException(
                status_code=400,
                detail=f"unknown {kind} setting {name!r}; allowed: {sorted(bounds)}",
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail=f"{kind} setting {name!r} must be numeric"
            ) from error
        low, high = bounds[name]
        if not (low <= number <= high):
            raise HTTPException(
                status_code=400,
                detail=f"{kind} setting {name!r} must be between {low} and {high}",
            )
        text = f"{number:g}"
        if name == "control_decimation":
            text = str(int(round(number)))
        cleaned.append((name, text))
    return cleaned


class JobManager:
    def __init__(self, runs_dir: Path, levels_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.levels_dir = levels_dir
        self.jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def active(self) -> TrainingJob | None:
        for job in self.jobs.values():
            if job.status in ("starting", "running"):
                return job
        return None

    def start(self, request: TrainRequest) -> TrainingJob:
        with self._lock:
            active = self.active()
            if active is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"job {active.job_id} is already running; stop it first",
                )
            catalog = LevelCatalog.load(self.levels_dir)
            if request.levels:
                unknown = [level for level in request.levels if level not in catalog.ids]
                if unknown:
                    raise HTTPException(status_code=400, detail=f"unknown levels: {unknown}")
                level_ids = list(request.levels)
            else:
                level_ids = [level.id for level in catalog.train_levels]
            curriculum = request.curriculum
            if len(level_ids) == 1:
                curriculum = "single"
            elif curriculum == "single":
                raise HTTPException(
                    status_code=400, detail="curriculum 'single' needs exactly one level"
                )
            stamp = time.strftime("%Y%m%d-%H%M%S")
            slug = re.sub(r"[^a-z0-9]+", "-", (request.name or "dashboard").lower()).strip("-")
            run_id = f"{stamp}-{slug or 'dashboard'}-{uuid.uuid4().hex[:4]}"
            command = [
                sys.executable,
                "-m",
                "parkour_lab.cli",
                "train",
                "--runs-dir",
                str(self.runs_dir),
                "--levels-dir",
                str(self.levels_dir),
                "--run-id",
                run_id,
                "--name",
                request.name or slug or "dashboard",
                "--curriculum",
                curriculum,
                "--steps",
                str(request.steps),
                "--envs",
                str(request.envs),
                "--workers",
                str(request.workers),
                "--seed",
                str(request.seed),
                *sum((["--level", level] for level in level_ids), []),
            ]
            if request.max_minutes is not None:
                command += ["--max-minutes", str(request.max_minutes)]
            if request.notes:
                command += ["--notes", request.notes]
            command += [
                "--rollout-steps",
                str(request.rollout_steps),
                "--hidden-dim",
                str(request.hidden_dim),
                "--eval-interval",
                str(request.eval_interval),
                "--checkpoint-interval",
                str(request.checkpoint_interval),
            ]
            for name, value in _bounded(request.settings, PHYSICS_BOUNDS, "training"):
                flag = {
                    "physics_timestep": "--physics-timestep",
                    "control_decimation": "--decimation",
                    "action_scale": "--action-scale",
                    "action_smoothing": "--action-smoothing",
                    "max_episode_seconds": "--max-episode-seconds",
                    "stagnation_seconds": "--stagnation-seconds",
                    "reward_ramp_fraction": "--reward-ramp-fraction",
                    "reward_scale": "--reward-scale",
                    "learning_rate": "--lr",
                    "entropy_coefficient": "--entropy",
                    "initial_log_std": "--init-log-std",
                }[name]
                command += [flag, value]
            for name, value in _bounded(request.reward, REWARD_BOUNDS, "reward"):
                command += ["--reward-set", f"{name}={value}"]
            log_dir = self.runs_dir / run_id / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            job = TrainingJob(
                job_id=f"job-{uuid.uuid4().hex[:8]}",
                run_id=run_id,
                command=command,
            )
            environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(
                command,
                cwd=str(_repo_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
            job.process = process
            job.status = "running"
            self.jobs[job.job_id] = job
            thread = threading.Thread(target=self._consume, args=(job,), daemon=True)
            thread.start()
            return job

    def _consume(self, job: TrainingJob) -> None:
        assert job.process is not None and job.process.stdout is not None
        log_path = self.runs_dir / job.run_id / "logs" / "train.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in job.process.stdout:
                job.log.append(line.rstrip("\n"))
                log_file.write(line)
                log_file.flush()
        job.exit_code = job.process.wait()
        job.status = "finished" if job.exit_code == 0 else "failed"

    def stop(self, job_id: str) -> TrainingJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job.process is not None and job.process.poll() is None:
            try:
                job.process.send_signal(signal.SIGINT)
            except ProcessLookupError:  # pragma: no cover
                pass
            try:
                job.process.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover
                job.process.terminate()
            job.status = "stopped"
            job.exit_code = job.process.returncode
        return job


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------
def create_app(
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    levels_dir: Path | str | None = None,
) -> FastAPI:
    runs_path = Path(runs_dir).resolve()
    levels_path = Path(levels_dir).resolve() if levels_dir else DEFAULT_LEVELS_DIR
    runs_path.mkdir(parents=True, exist_ok=True)
    manager = JobManager(runs_path, levels_path)

    app = FastAPI(title="AI Parkour Lab", version=__version__)
    app.state.runs_dir = runs_path
    app.state.levels_dir = levels_path
    app.state.jobs = manager

    def catalog() -> LevelCatalog:
        return LevelCatalog.load(levels_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        import mujoco
        import numpy

        return {
            "ok": True,
            "version": __version__,
            "runs_dir": str(runs_path),
            "levels_dir": str(levels_path),
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "mujoco": mujoco.__version__,
            "cpu_count": os.cpu_count(),
            "active_job": (manager.active().to_dict() if manager.active() else None),
        }

    @app.get("/api/levels")
    def levels() -> dict[str, Any]:
        try:
            level_catalog = catalog()
        except LevelError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "levels": [
                {
                    **level.to_dict(),
                    "difficulty_label": level.difficulty_label,
                    "spawn_support_z": level.spawn_support_z,
                }
                for level in level_catalog
            ]
        }

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        summaries = []
        for run in list_runs(runs_path):
            config = run.get("config", {}) or {}
            summary = run.get("summary", {}) or {}
            summaries.append(
                {
                    "run_id": run["run_id"],
                    "status": run.get("status", "unknown"),
                    "created_utc": run.get("created_utc"),
                    "updated_utc": run.get("updated_utc"),
                    "levels": config.get("level_ids", []),
                    "level_ids": config.get("level_ids", run.get("level_ids", [])),
                    "curriculum": config.get("curriculum") or run.get("curriculum"),
                    "total_steps": run.get("total_steps", summary.get("total_steps", 0)),
                    "best_eval_success": summary.get(
                        "best_eval_success", run.get("best_eval_success")
                    ),
                    "best_eval_completion": summary.get(
                        "best_eval_completion", run.get("best_eval_completion")
                    ),
                    "num_envs": config.get("num_envs", run.get("num_envs")),
                    "workers": config.get("workers", run.get("workers")),
                    "seed": config.get("seed", run.get("seed")),
                    "checkpoints": len(run.get("checkpoints", [])),
                    "replays": run.get("replay_count", 0),
                    "config": {
                        key: config.get(key)
                        for key in (
                            "num_envs",
                            "workers",
                            "seed",
                            "total_steps",
                            "action_scale",
                            "actuation",
                            "name",
                        )
                    },
                }
            )
        return {"runs": summaries}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        store = _store(runs_path, run_id)
        config = store.read_config()
        summary = store.read_summary()
        return {
            "run_id": run_id,
            "config": config,
            "summary": summary,
            "provenance": _CACHE.get(store.paths.provenance, lambda: read_json(store.paths.provenance, {})),
            "training": _training_rows(store),
            "episode_aggregate": _episode_aggregate(store),
            "evaluation_history": _evaluation_history(store),
            "latest_evaluation": _latest_evaluation(store),
            "checkpoints": store.checkpoints(),
            "replays": store.replays(),
        }

    @app.get("/api/runs/{run_id}/episodes")
    def run_episodes(
        run_id: str, limit: int = Query(200, ge=1, le=2000), level: str | None = None
    ) -> dict[str, Any]:
        store = _store(runs_path, run_id)
        records = []
        for record in store.episodes():
            if level and record.get("level_id") != level:
                continue
            records.append(record)
        return {"episodes": records[-limit:], "total": len(records)}

    @app.get("/api/runs/{run_id}/replays")
    def run_replays(run_id: str) -> dict[str, Any]:
        store = _store(runs_path, run_id)
        return {"replays": store.replays()}

    @app.get("/api/runs/{run_id}/replay/{name}")
    def replay(run_id: str, name: str) -> dict[str, Any]:
        store = _store(runs_path, run_id)
        if not name.endswith(".json.gz") or "/" in name or ".." in name:
            raise HTTPException(status_code=400, detail="invalid replay name")
        path = store.paths.replay_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="replay not found")
        return _replay_payload(path)

    @app.get("/api/compare")
    def compare(a: str, b: str) -> dict[str, Any]:
        from ..training.runner import compare_runs

        store_a = _store(runs_path, a)
        store_b = _store(runs_path, b)
        result = compare_runs(store_a, store_b)
        if "error" in result:
            raise HTTPException(status_code=409, detail=result["error"])
        return result

    @app.get("/api/benchmark")
    def benchmark(
        envs: str = "2,4", workers: int = 1, steps: int = 120, level: str | None = None
    ) -> dict[str, Any]:
        from ..env import EnvConfig
        from ..training.vector import benchmark_simulation

        level_catalog = catalog()
        level_ids = [level or level_catalog[0].id]
        counts = [int(value) for value in envs.split(",") if value.strip()][:4]
        results = []
        for count in counts:
            results.append(
                benchmark_simulation(
                    level_catalog,
                    levels=[level_ids[index % len(level_ids)] for index in range(count)],
                    num_envs=count,
                    steps=max(20, min(int(steps), 600)),
                    env_config=EnvConfig(level_id=level_ids[0]),
                )
            )
        return {"results": results}

    @app.post("/api/train")
    def start_training(request: TrainRequest) -> dict[str, Any]:
        job = manager.start(request)
        return job.to_dict()

    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in manager.jobs.values()]}

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, offset: int = 0) -> dict[str, Any]:
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job.status in ("starting", "running"):
            job_status = job.status
        else:
            job_status = job.status
        lines = list(job.log)
        return {
            "job": job.to_dict(),
            "lines": lines[max(0, offset) :],
            "offset": max(0, offset) + max(0, len(lines) - max(0, offset)),
            "status": job_status,
        }

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        return manager.stop(job_id).to_dict()

    @app.exception_handler(LevelError)
    def level_error_handler(_request: Any, exc: LevelError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg")

    return app


def run_dashboard(
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    host: str = "0.0.0.0",
    port: int = 8000,
    levels_dir: Path | str | None = None,
    open_browser: bool = False,
) -> int:
    import uvicorn

    app = create_app(runs_dir=runs_dir, levels_dir=levels_dir)
    if open_browser:  # pragma: no cover - convenience
        import threading
        import webbrowser

        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")
        ).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


__all__ = ["create_app", "run_dashboard"]
