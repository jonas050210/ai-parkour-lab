"""Fixed parkour level manifests: loading, schema validation and physics validation.

Levels are committed JSON documents validated by ``configs/levels/level.schema.json``.
Schema validation is necessary but not sufficient: :func:`validate_level_physics` also
checks that the spawn point actually has a supporting platform, that the finish is inside
the world bounds, that hazard volumes are above their support, and that no obstacle has
degenerate geometry.  A level that fails these checks is rejected before a run starts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .humanoid import PELVIS_HEIGHT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEVELS_DIR = ROOT / "configs" / "levels"
SCHEMA_PATH = DEFAULT_LEVELS_DIR / "level.schema.json"

OBSTACLE_KINDS = (
    "platform",
    "step",
    "gap",
    "spike_strip",
    "moving_platform",
    "wall",
    "rail",
    "rooftop",
)
# Kinds that create no collision geometry; they annotate a hole in the course.
NON_SOLID_KINDS = ("gap",)
SOLID_KINDS = tuple(kind for kind in OBSTACLE_KINDS if kind not in NON_SOLID_KINDS)

SPAWN_TOLERANCE = 0.06  # metres of allowed disagreement between manifest and support


class LevelError(ValueError):
    """Raised when a level manifest is invalid for simulation."""


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True, slots=True)
class Motion:
    axis: Vec3
    amplitude: float
    period_seconds: float


@dataclass(frozen=True, slots=True)
class Obstacle:
    id: str
    kind: str
    position: Vec3
    size: Vec3
    rotation_degrees: float = 0.0
    motion: Motion | None = None
    hazard: bool = False

    @property
    def top_z(self) -> float:
        return self.position.z + self.size.z

    @property
    def min_x(self) -> float:
        return self.position.x - self.size.x

    @property
    def max_x(self) -> float:
        return self.position.x + self.size.x

    @property
    def half_width(self) -> float:
        return self.size.y

    @property
    def solid(self) -> bool:
        return self.kind not in NON_SOLID_KINDS

    @property
    def hazard_volume(self) -> bool:
        return self.hazard

    def surface_at(self, x: float, y: float) -> float | None:
        """Top-surface height if (x, y) lies on this obstacle's footprint."""

        if not self.solid:
            return None
        if abs(x - self.position.x) > self.size.x:
            return None
        if abs(y - self.position.y) > self.size.y:
            return None
        return self.top_z


@dataclass(frozen=True, slots=True)
class Level:
    schema_version: int
    id: str
    title: str
    order: int
    summary: str
    spawn_position: Vec3
    spawn_yaw_degrees: float
    finish_position: Vec3
    finish_radius: float
    bounds_min: Vec3
    bounds_max: Vec3
    obstacles: tuple[Obstacle, ...]
    time_limit_seconds: float = 20.0
    difficulty: int = 1
    split: str = "train"
    tags: tuple[str, ...] = ()
    par_time_seconds: float | None = None
    source_path: Path | None = field(default=None, compare=False)

    # -- convenience -----------------------------------------------------------------------
    @property
    def width(self) -> float:
        return self.bounds_max.x - self.bounds_min.x

    @property
    def difficulty_label(self) -> str:
        return {1: "very easy", 2: "easy", 3: "medium", 4: "hard", 5: "expert"}.get(
            self.difficulty, "unknown"
        )

    @property
    def solid_obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(o for o in self.obstacles if o.solid)

    @property
    def hazard_obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(o for o in self.obstacles if o.hazard)

    @property
    def moving_obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(o for o in self.obstacles if o.motion is not None)

    @property
    def spawn_support_z(self) -> float:
        return self.support_height(self.spawn_position.x, self.spawn_position.y)

    def support_height(self, x: float, y: float) -> float:
        """Highest solid surface below/at (x, y); ``-inf`` means a bottomless gap."""

        best = -math.inf
        for obstacle in self.solid_obstacles:
            top = obstacle.surface_at(x, y)
            if top is not None and top > best:
                best = top
        return best

    def solid_obstacles_ahead(
        self, x: float, limit: int = 3
    ) -> tuple[Obstacle, ...]:
        ahead = [o for o in self.solid_obstacles if o.max_x >= x - 0.05]
        ahead.sort(key=lambda o: (o.min_x, o.id))
        return tuple(ahead[:limit])

    def hazards_ahead(self, x: float, limit: int = 2) -> tuple[Obstacle, ...]:
        ahead = [o for o in self.hazard_obstacles if o.max_x >= x - 0.05]
        ahead.sort(key=lambda o: (o.min_x, o.id))
        return tuple(ahead[:limit])

    def to_dict(self) -> dict[str, Any]:
        """Re-serialise to the manifest format (used for immutable run snapshots)."""

        def vec(v: Vec3) -> list[float]:
            return [v.x, v.y, v.z]

        obstacles: list[dict[str, Any]] = []
        for obstacle in self.obstacles:
            entry: dict[str, Any] = {
                "id": obstacle.id,
                "kind": obstacle.kind,
                "position": vec(obstacle.position),
                "size": vec(obstacle.size),
            }
            if obstacle.rotation_degrees:
                entry["rotation_degrees"] = obstacle.rotation_degrees
            if obstacle.motion is not None:
                entry["motion"] = {
                    "axis": vec(obstacle.motion.axis),
                    "amplitude": obstacle.motion.amplitude,
                    "period_seconds": obstacle.motion.period_seconds,
                }
            if obstacle.hazard:
                entry["hazard"] = True
            obstacles.append(entry)
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "order": self.order,
            "summary": self.summary,
            "difficulty": self.difficulty,
            "split": self.split,
            "tags": list(self.tags),
            "time_limit_seconds": self.time_limit_seconds,
            "par_time_seconds": self.par_time_seconds,
            "units": {
                "distance": "m",
                "angle": "degree",
                "up_axis": "+z",
                "forward_axis": "+x",
            },
            "spawn": {"position": vec(self.spawn_position), "yaw_degrees": self.spawn_yaw_degrees},
            "finish": {"position": vec(self.finish_position), "radius": self.finish_radius},
            "bounds": {"minimum": vec(self.bounds_min), "maximum": vec(self.bounds_max)},
            "obstacles": obstacles,
        }


def _vec3(value: Sequence[float], where: str) -> Vec3:
    if len(value) != 3:
        raise LevelError(f"{where} must have exactly three components")
    try:
        floats = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise LevelError(f"{where} must be numeric") from exc
    if any(not math.isfinite(v) for v in floats):
        raise LevelError(f"{where} must be finite")
    return Vec3(*floats)


def level_from_dict(data: dict[str, Any], *, source_path: Path | None = None) -> Level:
    """Parse a manifest dict into a :class:`Level` (no file or schema access)."""

    try:
        obstacles: list[Obstacle] = []
        for index, raw in enumerate(data["obstacles"]):
            kind = str(raw["kind"])
            if kind not in OBSTACLE_KINDS:
                raise LevelError(f"obstacle {index} has unknown kind {kind!r}")
            motion_raw = raw.get("motion")
            motion = None
            if motion_raw is not None:
                motion = Motion(
                    axis=_vec3(motion_raw["axis"], f"obstacles[{index}].motion.axis"),
                    amplitude=float(motion_raw["amplitude"]),
                    period_seconds=float(motion_raw["period_seconds"]),
                )
                if motion.amplitude < 0 or motion.period_seconds <= 0:
                    raise LevelError(f"obstacles[{index}].motion is degenerate")
            obstacles.append(
                Obstacle(
                    id=str(raw["id"]),
                    kind=kind,
                    position=_vec3(raw["position"], f"obstacles[{index}].position"),
                    size=_vec3(raw["size"], f"obstacles[{index}].size"),
                    rotation_degrees=float(raw.get("rotation_degrees", 0.0)),
                    motion=motion,
                    hazard=bool(raw.get("hazard", False)),
                )
            )
        level = Level(
            schema_version=int(data["schema_version"]),
            id=str(data["id"]),
            title=str(data["title"]),
            order=int(data["order"]),
            summary=str(data.get("summary", "")),
            spawn_position=_vec3(data["spawn"]["position"], "spawn.position"),
            spawn_yaw_degrees=float(data["spawn"].get("yaw_degrees", 0.0)),
            finish_position=_vec3(data["finish"]["position"], "finish.position"),
            finish_radius=float(data["finish"]["radius"]),
            bounds_min=_vec3(data["bounds"]["minimum"], "bounds.minimum"),
            bounds_max=_vec3(data["bounds"]["maximum"], "bounds.maximum"),
            obstacles=tuple(obstacles),
            time_limit_seconds=float(data.get("time_limit_seconds", 20.0)),
            difficulty=int(data.get("difficulty", 1)),
            split=str(data.get("split", "train")),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            par_time_seconds=(
                None if data.get("par_time_seconds") is None else float(data["par_time_seconds"])
            ),
            source_path=source_path,
        )
    except KeyError as exc:
        raise LevelError(f"level manifest missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, LevelError):
            raise
        raise LevelError(f"invalid level manifest value: {exc}") from exc
    validate_level_physics(level)
    return level


def validate_level_physics(level: Level) -> None:
    """Physics-level validation that a JSON Schema cannot express."""

    problems: list[str] = []

    if not level.id.strip():
        problems.append("id must not be empty")
    if not 1 <= level.difficulty <= 5:
        problems.append("difficulty must be between 1 and 5")
    if level.split not in ("train", "held-out"):
        problems.append("split must be 'train' or 'held-out'")
    if not 0.5 <= level.time_limit_seconds <= 120.0:
        problems.append("time_limit_seconds must be between 0.5 and 120")
    if level.finish_radius <= 0:
        problems.append("finish radius must be positive")
    if level.bounds_max.x <= level.bounds_min.x or level.bounds_max.y <= level.bounds_min.y:
        problems.append("bounds maximum must exceed minimum in x and y")
    if level.bounds_max.z <= level.bounds_min.z:
        problems.append("bounds maximum must exceed minimum in z")

    ids: set[str] = set()
    for obstacle in level.obstacles:
        if obstacle.id in ids:
            problems.append(f"duplicate obstacle id {obstacle.id!r}")
        ids.add(obstacle.id)
        for name, component in zip("xyz", obstacle.size.as_tuple()):
            if not math.isfinite(component) or component <= 0:
                problems.append(f"obstacle {obstacle.id!r} has non-positive size.{name}")
        if obstacle.motion is not None:
            axis = obstacle.motion.axis
            norm = math.sqrt(axis.x**2 + axis.y**2 + axis.z**2)
            if norm <= 1e-9:
                problems.append(f"obstacle {obstacle.id!r} motion axis is zero")
            if obstacle.size.z <= 0:
                problems.append(f"obstacle {obstacle.id!r} has no thickness")
        for corner_x, corner_y in (
            (obstacle.position.x - obstacle.size.x, obstacle.position.y - obstacle.size.y),
            (obstacle.position.x + obstacle.size.x, obstacle.position.y + obstacle.size.y),
        ):
            if not (
                level.bounds_min.x - 1e-6 <= corner_x <= level.bounds_max.x + 1e-6
                and level.bounds_min.y - 1e-6 <= corner_y <= level.bounds_max.y + 1e-6
            ):
                problems.append(
                    f"obstacle {obstacle.id!r} extends outside the world bounds"
                )
                break

    support = level.support_height(level.spawn_position.x, level.spawn_position.y)
    if not math.isfinite(support):
        problems.append("spawn point has no supporting platform below it")
    else:
        expected = support + PELVIS_HEIGHT
        if abs(expected - level.spawn_position.z) > SPAWN_TOLERANCE:
            problems.append(
                f"spawn z={level.spawn_position.z:.3f} must be about "
                f"support+pelvis height ({expected:.3f} +/- {SPAWN_TOLERANCE})"
            )

    if not (
        level.bounds_min.x <= level.finish_position.x <= level.bounds_max.x
        and level.bounds_min.y <= level.finish_position.y <= level.bounds_max.y
    ):
        problems.append("finish position lies outside the world bounds")
    finish_support = level.support_height(level.finish_position.x, level.finish_position.y)
    if not math.isfinite(finish_support):
        problems.append("finish position has no supporting platform below it")
    elif level.finish_position.z - finish_support > 2.5:
        problems.append("finish position is unrealistically far above its support")

    if problems:
        raise LevelError(f"level {level.id!r} failed physics validation: " + "; ".join(problems))


# --------------------------------------------------------------------------------------
# Loading and catalogues
# --------------------------------------------------------------------------------------


def load_schema(path: Path | None = None) -> dict[str, Any]:
    schema_path = Path(path) if path is not None else SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_against_schema(data: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    """Validate a manifest against the committed JSON Schema.

    ``jsonschema`` is a development dependency; the simulator does not require it, so a
    missing package only disables this optional pre-check (physics validation still runs).
    """

    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover - exercised only without the dev extra
        return []
    validator = Draft202012Validator(load_schema() if schema is None else schema)
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(data)
    ]


def load_level(path: str | Path, *, use_schema: bool = True) -> Level:
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LevelError(f"level file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise LevelError(f"level file {file_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LevelError(f"level file {file_path} must contain a JSON object")
    if use_schema:
        problems = validate_against_schema(data)
        if problems:
            raise LevelError(
                f"level {file_path.name} fails schema validation: " + "; ".join(problems)
            )
    return level_from_dict(data, source_path=file_path)


def level_paths(directory: str | Path | None = None) -> list[Path]:
    base = Path(directory) if directory is not None else DEFAULT_LEVELS_DIR
    return sorted(
        path for path in base.glob("level-*.json") if path.is_file()
    )


def load_levels(directory: str | Path | None = None, *, use_schema: bool = True) -> list[Level]:
    paths = level_paths(directory)
    if not paths:
        raise LevelError(f"no level manifests found in {directory or DEFAULT_LEVELS_DIR}")
    levels = [load_level(path, use_schema=use_schema) for path in paths]
    orders = [level.order for level in levels]
    if len(set(orders)) != len(orders):
        raise LevelError("level order values must be unique")
    ids = [level.id for level in levels]
    if len(set(ids)) != len(ids):
        raise LevelError("level ids must be unique")
    return sorted(levels, key=lambda level: level.order)


class LevelCatalog:
    """Ordered, validated collection of fixed levels."""

    def __init__(self, levels: Iterable[Level]) -> None:
        self._levels = tuple(sorted(levels, key=lambda level: level.order))
        if not self._levels:
            raise LevelError("a level catalog needs at least one level")
        self._by_id = {level.id: level for level in self._levels}
        if len(self._by_id) != len(self._levels):
            raise LevelError("duplicate level id in catalog")

    @classmethod
    def load(cls, directory: str | Path | None = None, *, use_schema: bool = True) -> "LevelCatalog":
        return cls(load_levels(directory, use_schema=use_schema))

    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterator[Level]:
        return iter(self._levels)

    def __getitem__(self, item: str | int) -> Level:
        if isinstance(item, int):
            return self._levels[item]
        try:
            return self._by_id[item]
        except KeyError as exc:
            raise LevelError(f"unknown level id: {item!r}") from exc

    @property
    def levels(self) -> tuple[Level, ...]:
        return self._levels

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(level.id for level in self._levels)

    @property
    def train_levels(self) -> tuple[Level, ...]:
        return tuple(level for level in self._levels if level.split == "train")

    @property
    def held_out_levels(self) -> tuple[Level, ...]:
        return tuple(level for level in self._levels if level.split == "held-out")

    def select(
        self,
        ids: Sequence[str] | None = None,
        *,
        split: str | None = None,
        max_difficulty: int | None = None,
        min_difficulty: int | None = None,
    ) -> tuple[Level, ...]:
        """Filter levels; ``ids`` preserves the caller's order."""

        if ids:
            return tuple(self[level_id] for level_id in ids)
        selected = self._levels
        if split is not None:
            selected = tuple(level for level in selected if level.split == split)
        if max_difficulty is not None:
            selected = tuple(level for level in selected if level.difficulty <= max_difficulty)
        if min_difficulty is not None:
            selected = tuple(level for level in selected if level.difficulty >= min_difficulty)
        if not selected:
            raise LevelError("level selection matched no levels")
        return selected


__all__ = [
    "DEFAULT_LEVELS_DIR",
    "Level",
    "LevelCatalog",
    "LevelError",
    "Motion",
    "NON_SOLID_KINDS",
    "OBSTACLE_KINDS",
    "Obstacle",
    "SOLID_KINDS",
    "Vec3",
    "level_from_dict",
    "level_paths",
    "load_level",
    "load_levels",
    "load_schema",
    "validate_against_schema",
    "validate_level_physics",
]
