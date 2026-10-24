"""Local, immutable run directories.

Layout of a run directory (all generated content is gitignored)::

    runs/<run-id>/
      config.json             exact resolved configuration of the run
      provenance.json         code revision, library versions, hardware, command
      summary.json            rolling status + final outcome
      metrics/training.csv    one row per training iteration
      metrics/episodes.jsonl  one JSON object per finished episode
      metrics/evaluation.csv  one row per evaluation sweep
      checkpoints/*.npz       policy + optimiser + normalisation + sampler state
      evaluation/*.json       full evaluation records
      replays/*.json.gz       recorded episode frames for the 3D viewer
      logs/                   captured console output (dashboard-launched runs)

Files are written atomically where a partial write would be misleading, and generated
names never overwrite an existing run directory.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT / "runs"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def git_revision(repo: Path = ROOT) -> dict[str, Any]:
    """Best-effort code revision without requiring a clean working tree."""

    info: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git not available
        pass
    return info


def _package_version(module_name: str) -> str:
    """Version of an installed distribution, without importing it.

    ``jsonschema.__version__`` is deprecated (and warns), so read installed metadata and
    fall back to ``__version__`` only for modules that are not registered as packages.
    """

    try:
        return importlib_metadata.version(module_name)
    except importlib_metadata.PackageNotFoundError:
        pass
    try:
        module = __import__(module_name)
    except ImportError:
        return "not installed"
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            return importlib_metadata.version(module.__name__)
        except importlib_metadata.PackageNotFoundError:
            return "unknown"
    return str(version)


def environment_provenance() -> dict[str, Any]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in ("numpy", "mujoco", "fastapi", "jsonschema"):
        try:
            __import__(module_name)
        except ImportError:
            versions[module_name] = "not installed"
        else:
            versions[module_name] = _package_version(module_name)
    cpu_count = os.cpu_count() or 1
    try:
        import torch  # type: ignore  # optional, only for GPU reporting

        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available = False
        gpu = None
    return {
        "timestamp_utc": utc_timestamp(),
        "python": versions["python"],
        "packages": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_count": cpu_count,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "mujoco_gl_backend": os.environ.get("MUJOCO_GL", "unset"),
        "command": " ".join(sys.argv),
        "git": git_revision(),
    }


@dataclass
class RunPaths:
    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def training_csv(self) -> Path:
        return self.metrics_dir / "training.csv"

    @property
    def episodes_jsonl(self) -> Path:
        return self.metrics_dir / "episodes.jsonl"

    @property
    def evaluation_csv(self) -> Path:
        return self.metrics_dir / "evaluation.csv"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def latest(self) -> Path:
        return self.checkpoint_dir / "latest.npz"

    @property
    def best(self) -> Path:
        return self.checkpoint_dir / "best.npz"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def replay_dir(self) -> Path:
        return self.root / "replays"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


class RunStore:
    """Create, append to and read immutable run records."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self._training_fields: list[str] | None = None

    # ----------------------------------------------------------------------------------
    # Creation
    # ----------------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        runs_dir: Path | str = DEFAULT_RUNS_DIR,
        *,
        run_id: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> "RunStore":
        base = Path(runs_dir)
        base.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            slug = _slug(name) if name else "run"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            run_id = f"{stamp}-{slug}"
            suffix = 1
            while (base / run_id).exists():
                suffix += 1
                run_id = f"{stamp}-{slug}-{suffix}"
        root = base / run_id
        if root.exists() and not resume:
            raise FileExistsError(
                f"run directory {root} already exists; choose another name or resume"
            )
        root.mkdir(parents=True, exist_ok=True)
        paths = RunPaths(root)
        for directory in (paths.metrics_dir, paths.checkpoint_dir, paths.evaluation_dir, paths.logs_dir, paths.replay_dir):
            directory.mkdir(parents=True, exist_ok=True)
        store = cls(paths)
        if not resume:
            if config is not None:
                atomic_write_json(paths.config, config)
            atomic_write_json(
                paths.provenance,
                provenance if provenance is not None else environment_provenance(),
            )
            atomic_write_json(
                paths.summary,
                {
                    "run_id": run_id,
                    "status": "initialised",
                    "created_utc": utc_timestamp(),
                    "updated_utc": utc_timestamp(),
                },
            )
        return store

    @classmethod
    def load(cls, root: Path | str) -> "RunStore":
        root = Path(root)
        if not (root / "config.json").exists():
            raise FileNotFoundError(f"{root} is not a run directory (missing config.json)")
        return cls(RunPaths(root))

    # ----------------------------------------------------------------------------------
    # Writing
    # ----------------------------------------------------------------------------------
    def append_episode(self, record: dict[str, Any]) -> None:
        self.paths.episodes_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.episodes_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")

    def append_training(self, row: dict[str, Any]) -> None:
        fields = list(row)
        if self._training_fields is None:
            self._training_fields = fields
            if self.paths.training_csv.exists():
                with self.paths.training_csv.open("r", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, None)
                    if header:
                        self._training_fields = header
        self._append_csv(self.paths.training_csv, row, self._training_fields)

    def append_evaluation(self, row: dict[str, Any]) -> None:
        self._append_csv(self.paths.evaluation_csv, row, list(row))

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})

    def update_summary(self, **values: Any) -> None:
        current = read_json(self.paths.summary, {}) or {}
        current.update(values)
        current["updated_utc"] = utc_timestamp()
        atomic_write_json(self.paths.summary, current)

    def write_json(self, relative: str, data: Any) -> Path:
        path = self.paths.root / relative
        atomic_write_json(path, _jsonable(data))
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.paths.root / relative
        atomic_write_text(path, text)
        return path

    def save_replay(self, name: str, payload: dict[str, Any]) -> Path:
        import gzip

        path = self.paths.replay_dir / (name if name.endswith(".json.gz") else f"{name}.json.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, separators=(",", ":"))
        tmp.replace(path)
        return path

    # ----------------------------------------------------------------------------------
    # Reading
    # ----------------------------------------------------------------------------------
    def read_config(self) -> dict[str, Any]:
        return read_json(self.paths.config, {}) or {}

    def read_summary(self) -> dict[str, Any]:
        return read_json(self.paths.summary, {}) or {}

    def episodes(self) -> Iterator[dict[str, Any]]:
        try:
            with self.paths.episodes_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return

    def training_rows(self) -> list[dict[str, Any]]:
        return _read_csv(self.paths.training_csv)

    def evaluation_rows(self) -> list[dict[str, Any]]:
        return _read_csv(self.paths.evaluation_csv)

    def evaluations(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.paths.evaluation_dir.glob("*.json")):
            data = read_json(path)
            if isinstance(data, dict):
                records.append(data)
        return records

    def checkpoints(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.paths.checkpoint_dir.glob("*.npz")):
            rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
        return rows

    def replays(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.paths.replay_dir.glob("*.json.gz")):
            rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        return rows


def pid_alive(pid: int | None) -> bool:
    """True if ``pid`` is a live process we are allowed to signal."""

    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def effective_status(
    summary: dict[str, Any],
    *,
    last_activity: float | None = None,
    now: float | None = None,
    stale_after_seconds: float = 900.0,
) -> str:
    """A run marked ``running`` whose process died is reported as ``interrupted``.

    Training processes are killed (Ctrl-C, OOM, a laptop suspending) often enough that a
    permanently "running" run would mislead both the CLI and the dashboard.

    Runs recorded before pid tracking existed have no ``pid``.  For those, ``last_activity``
    (the newest metrics-file mtime) decides: a run whose metrics have not moved for
    ``stale_after_seconds`` is also interrupted.
    """

    status = str(summary.get("status", "unknown"))
    if status != "running":
        return status
    if "pid" in summary:
        return "running" if pid_alive(summary.get("pid")) else "interrupted"
    if last_activity is None:
        return status
    reference = time.time() if now is None else now
    if reference - float(last_activity) > stale_after_seconds:
        return "interrupted"
    return status


def last_activity(path: Path) -> float | None:
    """Newest mtime among the artefacts a running trainer keeps rewriting."""

    newest: float | None = None
    candidates = [path / "summary.json"]
    metrics = path / "metrics"
    if metrics.exists():
        candidates.extend(metrics.glob("*"))
    for candidate in candidates:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return newest


def list_runs(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """Summarise every run directory, newest first."""

    base = Path(runs_dir)
    if not base.exists():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        config = read_json(child / "config.json")
        if config is None:
            continue
        summary = read_json(child / "summary.json", {}) or {}
        evaluations = []
        for path in sorted((child / "evaluation").glob("*.json")):
            data = read_json(path)
            if isinstance(data, dict):
                evaluations.append(data)
        best_eval = None
        if evaluations:
            best_eval = max(
                evaluations,
                key=lambda record: float(
                    (record.get("aggregate") or record.get("overall") or {}).get(
                        "success_rate", 0.0
                    )
                    if "aggregate" not in record
                    else record["aggregate"].get("overall", {}).get("success_rate", 0.0)
                ),
            )
        best_overall = (best_eval or {}).get("aggregate", {}).get("overall", {})
        # ``summary.total_steps`` is the *budget* recorded at startup, so a live run would
        # otherwise look finished.  The append-only training CSV holds the real progress.
        last_row = _last_csv_row(child / "metrics" / "training.csv")
        activity = last_activity(child)
        progress_steps = _as_int(last_row.get("total_steps"))
        runs.append(
            {
                "run_id": child.name,
                "path": str(child),
                "created_utc": summary.get("created_utc"),
                "updated_utc": summary.get("updated_utc"),
                "status": effective_status(summary, last_activity=activity),
                "recorded_status": summary.get("status", "unknown"),
                "total_steps": int(summary.get("total_steps", 0) or 0),
                "target_steps": int(config.get("total_steps", 0) or 0),
                "progress_steps": progress_steps,
                "best_eval_success": best_overall.get("success_rate"),
                "best_eval_completion": best_overall.get("mean_completion"),
                "curriculum": config.get("curriculum"),
                "num_envs": config.get("num_envs"),
                "workers": config.get("workers"),
                "seed": config.get("seed"),
                "level_ids": list(config.get("level_ids", [])),
                "config": config,
                "summary": summary,
                "best_evaluation": best_eval,
                "checkpoints": [p.name for p in (child / "checkpoints").glob("*.npz")]
                if (child / "checkpoints").exists()
                else [],
                "replay_count": len(list((child / "replays").glob("*.json.gz")))
                if (child / "replays").exists()
                else 0,
            }
        )
    runs.sort(key=lambda run: (run["created_utc"] or "", run["run_id"]), reverse=True)
    return runs


def delete_run(run_root: Path | str, *, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> None:
    """Delete one run directory (used by the dashboard's explicit delete action)."""

    root = Path(run_root).resolve()
    base = Path(runs_dir).resolve()
    if base not in root.parents:
        raise ValueError("refusing to delete a directory outside the runs directory")
    shutil.rmtree(root)


def _last_csv_row(path: Path) -> dict[str, Any]:
    """Return the last data row of an append-only CSV, or ``{}`` when it is missing."""

    last: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                last = dict(row)
    except FileNotFoundError:
        return {}
    return last


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError:
        return []


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def _slug(text: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in text.lower())
    return "-".join(part for part in cleaned.split("-") if part)[:48] or "run"


__all__ = [
    "DEFAULT_RUNS_DIR",
    "ROOT",
    "RunPaths",
    "RunStore",
    "atomic_write_json",
    "atomic_write_text",
    "delete_run",
    "environment_provenance",
    "git_revision",
    "effective_status",
    "last_activity",
    "list_runs",
    "pid_alive",
    "read_json",
    "utc_timestamp",
]
