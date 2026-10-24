"""Adaptive all-level training schedule.

The default schedule is an *adaptive* sampler that keeps training on every level while
spending most of its episodes on the weakest ones:

* each level keeps an exponentially weighted success rate,
* the sampling weight grows with the weakness of a level,
* a small novelty bonus makes sure every level is attempted,
* levels with no success and little progress are down-weighted so the learner is not
  stuck on an impossible wall,
* mastered levels are retested periodically (their weight decays with time since the
  last attempt but never reaches zero) to limit forgetting,
* the "frontier" limits how far ahead of the hardest solved level new levels are sampled,
  so the course order is respected without hard-locking the curriculum.

Everything is deterministic given the sampler seed, and the complete sampler state is
checkpointed so adaptive training resumes exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..levels import Level


@dataclass
class LevelPerformance:
    level_id: str
    order: int
    difficulty: int
    attempts: int = 0
    successes: int = 0
    ewma_success: float = 0.0
    mean_progress: float = 0.0
    best_progress: float = 0.0
    last_attempt_step: int = 0
    recent_returns: list[float] = field(default_factory=list)

    def record(self, success: bool, progress: float, episode_return: float, step: int, alpha: float) -> None:
        self.attempts += 1
        self.successes += int(bool(success))
        self.ewma_success = (1 - alpha) * self.ewma_success + alpha * float(bool(success))
        self.mean_progress = (1 - alpha) * self.mean_progress + alpha * float(progress)
        self.best_progress = max(self.best_progress, float(progress))
        self.last_attempt_step = int(step)
        self.recent_returns.append(float(episode_return))
        if len(self.recent_returns) > 25:
            self.recent_returns.pop(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "order": self.order,
            "difficulty": self.difficulty,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.successes / self.attempts if self.attempts else 0.0,
            "ewma_success": self.ewma_success,
            "mean_progress": self.mean_progress,
            "best_progress": self.best_progress,
            "last_attempt_step": self.last_attempt_step,
            "mean_return": float(np.mean(self.recent_returns)) if self.recent_returns else 0.0,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LevelPerformance":
        return cls(
            level_id=data["level_id"],
            order=int(data["order"]),
            difficulty=int(data["difficulty"]),
            attempts=int(data["attempts"]),
            successes=int(data["successes"]),
            ewma_success=float(data["ewma_success"]),
            mean_progress=float(data["mean_progress"]),
            best_progress=float(data["best_progress"]),
            last_attempt_step=int(data["last_attempt_step"]),
            recent_returns=list(data.get("recent_returns", [])),
        )


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Knobs for the adaptive sampler, all bounded and checkpointed."""

    ewma_alpha: float = 0.15
    weakness_scale: float = 3.0
    weakness_power: float = 1.5
    novelty_bonus: float = 2.0
    mastered_success_threshold: float = 0.75
    mastered_weight: float = 0.25
    stuck_attempts: int = 12
    stuck_progress_threshold: float = 0.15
    stuck_weight: float = 0.15
    frontier_window: int = 3
    exploration_fraction: float = 0.08

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CurriculumConfig":
        known = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown curriculum settings: {sorted(unknown)}")
        return cls(**known)


class LevelSampler:
    """Interface implemented by all schedules."""

    def sample(self, count: int) -> list[str]:  # pragma: no cover - protocol
        raise NotImplementedError

    def update(
        self, level_id: str, *, success: bool, progress: float, episode_return: float, step: int
    ) -> None: ...

    def stats(self) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class SingleLevelSampler(LevelSampler):
    """Always train one level (manual level selection)."""

    def __init__(self, level_id: str) -> None:
        self.level_id = level_id
        self.performance = {
            level_id: LevelPerformance(level_id, order=1, difficulty=1)
        }

    def sample(self, count: int) -> list[str]:
        return [self.level_id] * count

    def update(self, level_id: str, **kwargs: Any) -> None:
        performance = self.performance.setdefault(
            level_id, LevelPerformance(level_id, order=1, difficulty=1)
        )
        performance.attempts += 1
        performance.successes += int(bool(kwargs.get("success", False)))
        performance.ewma_success = (
            0.85 * performance.ewma_success + 0.15 * float(bool(kwargs.get("success", False)))
        )
        performance.mean_progress = 0.85 * performance.mean_progress + 0.15 * float(
            kwargs.get("progress", 0.0)
        )
        performance.last_attempt_step = int(kwargs.get("step", 0))

    def stats(self) -> dict[str, Any]:
        return {"mode": "single", "levels": {k: v.to_dict() for k, v in self.performance.items()}}

    def state_dict(self) -> dict[str, Any]:
        return {"mode": "single", "performance": {k: v.to_dict() for k, v in self.performance.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, value in state.get("performance", {}).items():
            self.performance[key] = LevelPerformance.from_dict(value)


class UniformLevelSampler(LevelSampler):
    """Uniform random level selection (used as a control condition)."""

    def __init__(self, levels: Sequence[Level], seed: int = 0) -> None:
        if not levels:
            raise ValueError("uniform sampler needs at least one level")
        self.levels = tuple(levels)
        self.rng = np.random.default_rng(seed)
        self.performance = {
            level.id: LevelPerformance(level.id, level.order, level.difficulty)
            for level in levels
        }

    def sample(self, count: int) -> list[str]:
        ids = [level.id for level in self.levels]
        indices = self.rng.integers(0, len(ids), size=count)
        return [ids[int(index)] for index in indices]

    def update(self, level_id: str, **kwargs: Any) -> None:
        performance = self.performance.get(level_id)
        if performance is None:
            return
        performance.record(
            bool(kwargs.get("success", False)),
            float(kwargs.get("progress", 0.0)),
            float(kwargs.get("episode_return", 0.0)),
            int(kwargs.get("step", 0)),
            alpha=0.15,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "mode": "uniform",
            "levels": {key: value.to_dict() for key, value in self.performance.items()},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "uniform",
            "rng_state": self.rng.bit_generator.state,
            "performance": {key: value.to_dict() for key, value in self.performance.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        for key, value in state.get("performance", {}).items():
            self.performance[key] = LevelPerformance.from_dict(value)


class AdaptiveLevelSampler(LevelSampler):
    """Weakness-driven, novelty-aware, periodically retesting schedule."""

    def __init__(
        self,
        levels: Sequence[Level],
        *,
        seed: int = 0,
        config: CurriculumConfig | None = None,
    ) -> None:
        if not levels:
            raise ValueError("adaptive sampler needs at least one level")
        self.levels = tuple(sorted(levels, key=lambda level: level.order))
        self.config = config or CurriculumConfig()
        self.rng = np.random.default_rng(seed)
        self.performance: dict[str, LevelPerformance] = {
            level.id: LevelPerformance(level.id, level.order, level.difficulty)
            for level in self.levels
        }
        self._order = {level.id: level.order for level in self.levels}
        self._max_order = max(self._order.values())
        self._last_weights: dict[str, float] = {}

    # ----------------------------------------------------------------------------------
    def _frontier_order(self) -> int:
        """Highest order the sampler may explore, from solved levels plus a window.

        The first never-attempted level is always included so the course order is
        respected even before anything is solved.
        """

        solved = 1
        for level in self.levels:
            performance = self.performance[level.id]
            if performance.attempts > 0 and performance.ewma_success >= 0.15:
                solved = max(solved, level.order)
        unattempted = [
            level.order for level in self.levels if self.performance[level.id].attempts == 0
        ]
        if unattempted:
            solved = max(solved, min(unattempted))
        return min(self._max_order, solved + self.config.frontier_window)

    def weights(self, step: int | None = None) -> dict[str, float]:
        cfg = self.config
        frontier = self._frontier_order()
        weights: dict[str, float] = {}
        for level in self.levels:
            p = self.performance[level.id]
            if p.order > frontier:
                weights[level.id] = 0.0
                continue
            if p.attempts == 0:
                weight = 1.0 + cfg.novelty_bonus
            else:
                weakness = (1.0 - min(max(p.ewma_success, 0.0), 1.0)) ** cfg.weakness_power
                weight = 0.35 + cfg.weakness_scale * weakness
                if p.ewma_success >= cfg.mastered_success_threshold:
                    # Retest mastered levels: weight recovers with time since last attempt.
                    gap = 1.0
                    if step is not None:
                        gap = 1.0 + min(2.0, (step - p.last_attempt_step) / 20000.0)
                    weight = cfg.mastered_weight * gap
                if (
                    p.attempts >= cfg.stuck_attempts
                    and p.ewma_success < 0.05
                    and p.mean_progress < cfg.stuck_progress_threshold
                ):
                    weight *= cfg.stuck_weight
            weights[level.id] = max(weight, 1e-3)
        total = sum(weights.values())
        if total <= 0:  # pragma: no cover - defensive
            return {level.id: 1.0 for level in self.levels}
        return {key: value / total for key, value in weights.items()}

    def sample(self, count: int) -> list[str]:
        step = max(
            (p.last_attempt_step for p in self.performance.values()), default=0
        )
        weights = self.weights(step)
        ids = list(weights)
        probabilities = np.array([weights[key] for key in ids], dtype=np.float64)
        probabilities /= probabilities.sum()
        samples = self.rng.choice(ids, size=count, p=probabilities).tolist()
        if self.config.exploration_fraction > 0:
            # Explore *inside* the allowed set.  Drawing uniformly from all levels would
            # bypass the frontier and send a beginner policy to the hardest course.
            allowed = [key for key in ids if weights[key] > 0.0]
            uniforms = self.rng.random(count) < self.config.exploration_fraction
            for index, explore in enumerate(uniforms):
                if explore and allowed:
                    samples[index] = str(self.rng.choice(allowed))
        return [str(sample) for sample in samples]

    def update(
        self, level_id: str, *, success: bool, progress: float, episode_return: float, step: int
    ) -> None:
        performance = self.performance.get(level_id)
        if performance is None:
            return
        performance.record(
            success,
            progress,
            episode_return,
            step,
            alpha=self.config.ewma_alpha,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "mode": "adaptive",
            "frontier_order": self._frontier_order(),
            "weights": self.weights(),
            "levels": {key: value.to_dict() for key, value in self.performance.items()},
            "config": self.config.to_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "adaptive",
            "rng_state": self.rng.bit_generator.state,
            "performance": {
                key: {
                    **value.to_dict(),
                    "recent_returns": list(value.recent_returns),
                }
                for key, value in self.performance.items()
            },
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        if "config" in state:
            self.config = CurriculumConfig.from_dict(state["config"])
        for key, value in state.get("performance", {}).items():
            self.performance[key] = LevelPerformance.from_dict(value)


def make_sampler(
    levels: Sequence[Level],
    *,
    mode: str = "adaptive",
    seed: int = 0,
    config: CurriculumConfig | None = None,
) -> LevelSampler:
    """Build a sampler by name: ``adaptive``, ``uniform`` or ``single``."""

    if not levels:
        raise ValueError("at least one level is required")
    if mode == "single":
        return SingleLevelSampler(levels[0].id)
    if mode == "uniform":
        return UniformLevelSampler(levels, seed=seed)
    if mode == "adaptive":
        return AdaptiveLevelSampler(levels, seed=seed, config=config)
    raise ValueError(f"unknown curriculum mode: {mode!r}")


def summarise_performance(sampler: LevelSampler) -> list[dict[str, Any]]:
    stats = sampler.stats()
    rows = []
    for level_id, data in stats.get("levels", {}).items():
        rows.append({"level_id": level_id, **{k: v for k, v in data.items() if k != "level_id"}})
    return sorted(rows, key=lambda row: row.get("order", 0))


__all__ = [
    "AdaptiveLevelSampler",
    "CurriculumConfig",
    "LevelPerformance",
    "LevelSampler",
    "SingleLevelSampler",
    "UniformLevelSampler",
    "make_sampler",
    "summarise_performance",
]
