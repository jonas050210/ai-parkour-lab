"""Explain *why* a policy fails, from one traced rollout per seed.

Training metrics say whether a run improved; they do not say what the robot is doing.  A
5 M-step run can sit at 59 % completion with a healthy value function and no success, and
the difference between "it hops", "it shuffles" and "it walks and then trips on the step"
is invisible in an aggregate.  This module runs a handful of episodes with the same
observation/action path as training and evaluation, keeps a small trace (forward speed,
pelvis height, per-foot contact) and turns it into named findings.

Everything here is read-only: no weights are updated, no reward is touched, and the
environment is the shared :class:`~parkour_lab.env.ParkourEnv`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from .env import EnvConfig, ParkourEnv
from .levels import Level
from .policies import ActorCritic

#: A trajectory counts as "standing" below this mean forward speed (m/s).
STANDING_SPEED = 0.25
#: Above this fraction of control steps with both feet off the ground, the gait is a hop.
HOPPING_FLIGHT_FRACTION = 0.15
#: Completion below this reads as "did not leave the start".
STUCK_COMPLETION = 0.15


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Compact summary of one traced episode."""

    seed: int
    level_id: str
    success: bool
    termination_reason: str
    steps: int
    seconds: float
    completion: float
    distance: float
    max_distance: float
    mean_forward_speed: float
    peak_forward_speed: float
    flight_fraction: float
    contact_switches: int
    final_pelvis_height: float
    min_pelvis_height: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceAccumulator:
    """Running quantities of one episode; kept separate so the loop stays readable."""

    speed_sum: float = 0.0
    peak_speed: float = 0.0
    flight_steps: int = 0
    contact_switches: int = 0
    min_height: float = math.inf
    _previous_load: tuple[bool, bool] | None = field(default=None, repr=False)

    def update(self, info: dict[str, Any]) -> None:
        speed = float(info.get("forward_speed", 0.0))
        self.speed_sum += speed
        self.peak_speed = max(self.peak_speed, speed)
        left = bool(info.get("left_contact", False))
        right = bool(info.get("right_contact", False))
        if not left and not right:
            self.flight_steps += 1
        if self._previous_load is not None and (left, right) != self._previous_load:
            # A step is a change of which feet bear load (touch-down or lift-off).
            self.contact_switches += 1
        self._previous_load = (left, right)
        self.min_height = min(self.min_height, float(info.get("pelvis_height", math.inf)))


def trace_episode(
    policy: ActorCritic,
    level: Level,
    *,
    seed: int,
    env_config: EnvConfig | None = None,
    deterministic: bool = True,
    randomize_reset: bool = True,
    rng: np.random.Generator | None = None,
) -> EpisodeTrace:
    """Run one episode and keep the quantities that explain the outcome."""

    base = env_config or EnvConfig(level_id=level.id)
    config = replace(
        base,
        level_id=level.id,
        seed=seed,
        randomize_reset=randomize_reset,
        record_frames=False,
    )
    env = ParkourEnv(level, config)
    generator = rng if rng is not None else np.random.default_rng(seed + 1)
    accumulator = TraceAccumulator()
    try:
        observation, _ = env.reset(seed=seed, options={"deterministic": not config.randomize_reset})
        start_x = float(env.data.qpos[0])
        max_x = start_x
        steps = 0
        while True:
            action = policy.act(observation, deterministic=deterministic, rng=generator)
            result = env.step(action)
            observation = result.observation
            steps += 1
            max_x = max(max_x, float(env.data.qpos[0]))
            accumulator.update(result.info)
            if result.terminated or result.truncated:
                break
        summary = env.episode_summary()
        final_x = float(env.data.qpos[0])
    finally:
        env.close()
    seconds = steps * config.control_dt
    return EpisodeTrace(
        seed=int(seed),
        level_id=level.id,
        success=bool(summary["success"]),
        termination_reason=str(summary["termination_reason"]),
        steps=steps,
        seconds=seconds,
        completion=float(summary["completion_fraction"]),
        distance=float(final_x - start_x),
        max_distance=float(max_x - start_x),
        mean_forward_speed=accumulator.speed_sum / max(1, steps),
        peak_forward_speed=accumulator.peak_speed,
        flight_fraction=accumulator.flight_steps / max(1, steps),
        contact_switches=accumulator.contact_switches,
        final_pelvis_height=float(summary["metrics"].get("final_height", 0.0)),
        min_pelvis_height=(
            float(accumulator.min_height) if math.isfinite(accumulator.min_height) else 0.0
        ),
    )


def _finding(name: str, severity: str, message: str) -> dict[str, str]:
    return {"name": name, "severity": severity, "message": message}


def summarise_traces(traces: Sequence[EpisodeTrace]) -> dict[str, Any]:
    """Aggregate traces and turn them into named, human-readable findings."""

    if not traces:
        raise ValueError("at least one trace is required")
    count = len(traces)
    successes = sum(1 for trace in traces if trace.success)
    completions = [trace.completion for trace in traces]
    terminations: dict[str, int] = {}
    for trace in traces:
        terminations[trace.termination_reason] = terminations.get(trace.termination_reason, 0) + 1
    dominant = max(terminations.items(), key=lambda item: item[1])
    mean_completion = float(np.mean(completions))
    mean_distance = float(np.mean([trace.distance for trace in traces]))
    mean_speed = float(np.mean([trace.mean_forward_speed for trace in traces]))
    flight = float(np.mean([trace.flight_fraction for trace in traces]))
    switches = float(np.mean([trace.contact_switches for trace in traces]))
    spread = float(max(completions) - min(completions))
    success_rate = successes / count

    findings: list[dict[str, str]] = []
    if success_rate >= 0.5:
        findings.append(
            _finding(
                "success",
                "info",
                f"{successes}/{count} episodes reach the finish "
                f"(mean {mean_completion * 100:.0f} % completion, {mean_speed:.2f} m/s).",
            )
        )
    else:
        findings.append(
            _finding(
                "no_success",
                "warning",
                f"0/{count} episodes reach the finish; dominant ending is "
                f"'{dominant[0]}' ({dominant[1]}/{count}).",
            )
        )
    if flight > HOPPING_FLIGHT_FRACTION:
        findings.append(
            _finding(
                "hopping",
                "warning",
                f"both feet are airborne for {flight * 100:.0f} % of control steps: "
                "the policy hops rather than walks.",
            )
        )
    if flight <= HOPPING_FLIGHT_FRACTION and mean_speed < STANDING_SPEED:
        findings.append(
            _finding(
                "shuffling",
                "warning",
                f"mean forward speed is {mean_speed:.2f} m/s with a {flight * 100:.0f} % flight "
                "phase: the policy mostly stands or shuffles in place.",
            )
        )
    if mean_completion < STUCK_COMPLETION:
        findings.append(
            _finding(
                "stuck_near_start",
                "warning",
                f"episodes end after {mean_distance:.2f} m of forward travel on average, which "
                f"is {mean_completion * 100:.0f} % of the course.",
            )
        )
    if switches < 1.0:
        findings.append(
            _finding(
                "no_weight_transfer",
                "warning",
                f"{switches:.1f} load changes between feet per episode: the policy never puts "
                "weight on one foot and lifts the other.",
            )
        )
    if spread > 0.3 and not successes:
        worst = min(traces, key=lambda trace: trace.completion)
        findings.append(
            _finding(
                "seed_sensitivity",
                "info",
                f"completion ranges {min(completions) * 100:.0f}–{max(completions) * 100:.0f} % "
                f"across seeds; the worst is seed {worst.seed} "
                f"({worst.termination_reason}, {worst.distance:.2f} m).",
            )
        )
    collapses = [trace for trace in traces if trace.termination_reason in ("collapsed", "fell_over")]
    if collapses:
        mean_height = float(np.mean([trace.final_pelvis_height for trace in collapses]))
        findings.append(
            _finding(
                "collapse_height",
                "info",
                f"{len(collapses)}/{count} episodes end on the ground at a pelvis height of "
                f"{mean_height:.2f} m.",
            )
        )
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": success_rate,
        "mean_completion": mean_completion,
        "mean_distance": mean_distance,
        "mean_forward_speed": mean_speed,
        "flight_fraction": flight,
        "contact_switches": switches,
        "completion_spread": spread,
        "terminations": terminations,
        "findings": findings,
        "traces": [trace.to_dict() for trace in traces],
    }


def diagnose_policy(
    policy: ActorCritic,
    level: Level,
    *,
    seeds: Sequence[int],
    env_config: EnvConfig | None = None,
    deterministic: bool = True,
    randomize_reset: bool = True,
) -> dict[str, Any]:
    """Trace one episode per seed and summarise what the policy is doing."""

    traces = [
        trace_episode(
            policy,
            level,
            seed=int(seed),
            env_config=env_config,
            deterministic=deterministic,
            randomize_reset=randomize_reset,
        )
        for seed in seeds
    ]
    summary = summarise_traces(traces)
    summary["level_id"] = level.id
    summary["level_title"] = level.title
    summary["deterministic"] = deterministic
    summary["randomize_reset"] = randomize_reset
    return summary


def format_diagnosis(diagnosis: dict[str, Any]) -> str:
    """Render a diagnosis as a short text report for the terminal."""

    lines = [
        f"{diagnosis['level_id']} ({diagnosis['level_title']}): "
        f"{diagnosis['successes']}/{diagnosis['episodes']} episodes complete, "
        f"mean completion {diagnosis['mean_completion'] * 100:.1f} %, "
        f"mean speed {diagnosis['mean_forward_speed']:.2f} m/s",
    ]
    header = f"{'seed':>6}  {'completion':>10}  {'distance':>8}  {'speed':>7}  {'flight':>7}  {'end':<14}"
    lines.append(header)
    lines.append("-" * len(header))
    for trace in diagnosis["traces"]:
        lines.append(
            f"{trace['seed']:>6}  {trace['completion'] * 100:>9.1f}%  "
            f"{trace['distance']:>7.2f}m  {trace['mean_forward_speed']:>6.2f} "
            f"{trace['flight_fraction'] * 100:>6.0f}%  {trace['termination_reason']:<14}"
        )
    lines.append("")
    for finding in diagnosis["findings"]:
        marker = {"warning": "!", "info": "·", "ok": "+"}.get(finding["severity"], "·")
        lines.append(f"{marker} {finding['message']}")
    return "\n".join(lines)


__all__ = [
    "EpisodeTrace",
    "TraceAccumulator",
    "diagnose_policy",
    "format_diagnosis",
    "summarise_traces",
    "trace_episode",
]
