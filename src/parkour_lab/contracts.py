"""Small, dependency-free interfaces shared by future simulator and research components.

These types specify boundaries only. Concrete physics, learning, logging and storage code is not
part of the research foundation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

Observation: TypeAlias = Any
Action: TypeAlias = Any
Info: TypeAlias = dict[str, Any]


class Environment(Protocol):
    """Gymnasium-shaped boundary for one episode environment or vector adapter."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, Info]:
        """Reset an episode and return its initial observation and diagnostic info."""
        ...

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, Info]:
        """Advance once: observation, reward, terminated, truncated, diagnostic info."""
        ...

    def close(self) -> None:
        """Release backend resources, if any."""
        ...


class Policy(Protocol):
    """Inference boundary; learning algorithms and policy implementations remain separate."""

    def act(self, observation: Observation, *, deterministic: bool = False) -> Action:
        """Map one observation to a bounded environment action."""
        ...


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Portable episode-level evaluation record, independent of the physics backend."""

    run_id: str
    level_id: str
    seed: int
    success: bool
    completion_fraction: float
    episode_return: float
    sim_steps: int
    termination_reason: str
    elapsed_seconds: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.level_id.strip():
            raise ValueError("level_id must not be empty")
        if (
            not math.isfinite(self.completion_fraction)
            or not 0.0 <= self.completion_fraction <= 1.0
        ):
            raise ValueError("completion_fraction must be finite and between 0 and 1")
        if not math.isfinite(self.episode_return):
            raise ValueError("episode_return must be finite")
        if self.sim_steps < 0:
            raise ValueError("sim_steps must not be negative")
        if self.elapsed_seconds is not None and (
            not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0
        ):
            raise ValueError("elapsed_seconds must be finite and not negative")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("episode metrics must be finite")
