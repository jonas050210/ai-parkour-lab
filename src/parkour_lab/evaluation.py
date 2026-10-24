"""Deterministic evaluation and comparison of trained policies.

Evaluation never updates policy weights, never touches the training sampler, and always
runs the frozen level manifests.  Each ``(level, seed)`` pair produces one record with
success, completion, termination reason, duration and safety/efficiency metrics; records
are aggregated per level and overall so two checkpoints can be compared on matched seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from .env import EnvConfig, ParkourEnv
from .levels import Level, LevelCatalog
from .policies import ActorCritic

DEFAULT_EVAL_SEEDS = (11, 23, 37)


def expand_eval_seeds(seeds: Sequence[int], count: int | None = None) -> list[int]:
    """Return exactly ``count`` distinct evaluation seeds.

    Asking for more episodes than there are configured seeds used to silently return fewer
    episodes, which made ``--eval-episodes`` a lie and weakened every reported rate.  Extra
    seeds are derived deterministically from the highest configured seed, so evaluation stays
    reproducible and no two episodes share a reset.
    """

    base = [int(seed) for seed in seeds]
    if not base:
        raise ValueError("at least one evaluation seed is required")
    if count is None:
        return base
    count = int(count)
    if count <= 0:
        raise ValueError("episode count must be positive")
    if count <= len(base):
        return base[:count]
    highest = max(base)
    extra = [highest + 1000 * (index + 1) for index in range(count - len(base))]
    return base + extra


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """How a policy is evaluated.

    ``deterministic`` controls the action distribution (mean vs. sample); ``randomize_reset``
    controls whether each seed also perturbs the initial state.  With both a deterministic
    policy *and* a fixed initial state every seed plays the identical episode, which makes a
    multi-seed evaluation look artificially stable; keeping reset noise on means the seeds
    measure genuine robustness to where the robot starts.  Both are reproducible: the reset
    noise is drawn from a generator seeded by ``seed``.
    """

    seeds: tuple[int, ...] = DEFAULT_EVAL_SEEDS
    deterministic: bool = True
    randomize_reset: bool = True
    max_episodes_per_level: int | None = None
    record_frames: bool = False
    frame_stride: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "deterministic": self.deterministic,
            "randomize_reset": self.randomize_reset,
            "max_episodes_per_level": self.max_episodes_per_level,
            "record_frames": self.record_frames,
            "frame_stride": self.frame_stride,
        }


def rollout_episode(
    policy: ActorCritic,
    level: Level,
    *,
    seed: int,
    env_config: EnvConfig | None = None,
    deterministic: bool = True,
    record_frames: bool = False,
    frame_stride: int = 2,
    randomize_reset: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run one full episode and return a serialisable record."""

    config = env_config or EnvConfig(level_id=level.id)
    # Evaluation must reproduce the training dynamics exactly (same action scaling,
    # smoothing, timestep and decimation).  Only the reset distribution is a choice.
    config = replace(
        config,
        level_id=level.id,
        seed=seed,
        randomize_reset=randomize_reset,
        record_frames=record_frames or config.record_frames,
        frame_stride=frame_stride,
    )
    env = ParkourEnv(level, config)
    generator = rng if rng is not None else np.random.default_rng(seed + 1)
    observation, info = env.reset(
        seed=seed, options={"deterministic": not randomize_reset}
    )
    while True:
        action = policy.act(observation, deterministic=deterministic, rng=generator)
        result = env.step(action)
        # Closed loop: the policy must act on the state it just reached.  Forgetting this
        # assignment left every evaluation episode acting on the reset observation, which
        # silently produced much worse (and meaningless) evaluation numbers.
        observation = result.observation
        if result.terminated or result.truncated:
            break
    summary = env.episode_summary()
    summary["level_title"] = level.title
    summary["difficulty"] = level.difficulty
    summary["frames"] = list(env.frames) if record_frames else []
    summary["initial_info"] = {
        "support_height": info.get("support_height"),
        "time_limit_seconds": info.get("time_limit_seconds"),
    }
    env.close()
    return summary


def evaluate_policy(
    policy: ActorCritic,
    catalog: LevelCatalog,
    level_ids: Sequence[str],
    *,
    config: EvalConfig | None = None,
    env_config: EnvConfig | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Evaluate a policy on the given levels and seeds without training state changes."""

    cfg = config or EvalConfig()
    records: list[dict[str, Any]] = []
    total_episodes = len(level_ids) * len(cfg.seeds)
    completed = 0
    for level_id in level_ids:
        level = catalog[level_id]
        seeds = expand_eval_seeds(cfg.seeds, cfg.max_episodes_per_level)
        for seed in seeds:
            record = rollout_episode(
                policy,
                level,
                seed=seed,
                env_config=env_config,
                deterministic=cfg.deterministic,
                record_frames=cfg.record_frames,
                frame_stride=cfg.frame_stride,
                randomize_reset=cfg.randomize_reset,
            )
            record["seed"] = seed
            records.append(record)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_episodes, record)
    return {
        "config": cfg.to_dict(),
        "levels": list(level_ids),
        "records": records,
        "aggregate": aggregate_records(records),
    }


def evaluate_episode_set(
    policy: ActorCritic,
    catalog: LevelCatalog,
    level_ids: Sequence[str],
    *,
    seeds: Sequence[int],
    env_config: EnvConfig | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    return evaluate_policy(
        policy,
        catalog,
        level_ids,
        config=EvalConfig(seeds=tuple(seeds), deterministic=deterministic),
        env_config=env_config,
    )


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode records overall and per level."""

    if not records:
        return {
            "overall": {
                "episodes": 0,
                "success_rate": 0.0,
                "mean_completion": 0.0,
                "mean_return": 0.0,
                "mean_duration": 0.0,
            },
            "per_level": {},
            "terminations": {},
        }
    per_level: dict[str, dict[str, Any]] = {}
    terminations: dict[str, int] = {}
    for record in records:
        level_id = record.get("level_id", "unknown")
        bucket = per_level.setdefault(
            level_id,
            {
                "episodes": 0,
                "successes": 0,
                "completions": [],
                "returns": [],
                "durations": [],
                "terminations": {},
                "peak_contact_force": [],
                "energy": [],
            },
        )
        bucket["episodes"] += 1
        bucket["successes"] += int(bool(record.get("success", False)))
        bucket["completions"].append(float(record.get("completion_fraction", 0.0)))
        bucket["returns"].append(float(record.get("episode_return", 0.0)))
        bucket["durations"].append(float(record.get("elapsed_seconds", 0.0) or 0.0))
        metrics = record.get("metrics", {}) or {}
        if "max_contact_force" in metrics:
            bucket["peak_contact_force"].append(float(metrics["max_contact_force"]))
        if "energy" in metrics:
            bucket["energy"].append(float(metrics["energy"]))
        reason = str(record.get("termination_reason", "unknown"))
        bucket["terminations"][reason] = bucket["terminations"].get(reason, 0) + 1
        terminations[reason] = terminations.get(reason, 0) + 1

    per_level_summary: dict[str, dict[str, Any]] = {}
    for level_id, bucket in per_level.items():
        episodes = bucket["episodes"]
        per_level_summary[level_id] = {
            "episodes": episodes,
            "successes": bucket["successes"],
            "success_rate": bucket["successes"] / episodes,
            "mean_completion": float(np.mean(bucket["completions"])),
            "std_completion": float(np.std(bucket["completions"])),
            "mean_return": float(np.mean(bucket["returns"])),
            "mean_duration": float(np.mean(bucket["durations"])),
            "peak_contact_force": float(np.max(bucket["peak_contact_force"]))
            if bucket["peak_contact_force"]
            else 0.0,
            "mean_energy": float(np.mean(bucket["energy"])) if bucket["energy"] else 0.0,
            "terminations": bucket["terminations"],
        }
    completions = [float(record.get("completion_fraction", 0.0)) for record in records]
    returns = [float(record.get("episode_return", 0.0)) for record in records]
    durations = [float(record.get("elapsed_seconds", 0.0) or 0.0) for record in records]
    successes = sum(int(bool(record.get("success", False))) for record in records)
    return {
        "overall": {
            "episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
            "mean_completion": float(np.mean(completions)),
            "std_completion": float(np.std(completions)),
            "mean_return": float(np.mean(returns)),
            "mean_duration": float(np.mean(durations)),
        },
        "per_level": per_level_summary,
        "terminations": terminations,
    }


def compare_aggregates(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Paired per-level comparison of two evaluation aggregates."""

    rows: list[dict[str, Any]] = []
    base_levels = baseline.get("per_level", {})
    cand_levels = candidate.get("per_level", {})
    for level_id in sorted(set(base_levels) | set(cand_levels)):
        base = base_levels.get(level_id, {})
        cand = cand_levels.get(level_id, {})
        rows.append(
            {
                "level_id": level_id,
                "baseline_success": float(base.get("success_rate", 0.0)),
                "candidate_success": float(cand.get("success_rate", 0.0)),
                "delta_success": float(cand.get("success_rate", 0.0))
                - float(base.get("success_rate", 0.0)),
                "baseline_completion": float(base.get("mean_completion", 0.0)),
                "candidate_completion": float(cand.get("mean_completion", 0.0)),
                "delta_completion": float(cand.get("mean_completion", 0.0))
                - float(base.get("mean_completion", 0.0)),
                "episodes": min(int(base.get("episodes", 0)), int(cand.get("episodes", 0))),
            }
        )
    base_overall = baseline.get("overall", {})
    cand_overall = candidate.get("overall", {})
    return {
        "rows": rows,
        "overall": {
            "baseline_success_rate": float(base_overall.get("success_rate", 0.0)),
            "candidate_success_rate": float(cand_overall.get("success_rate", 0.0)),
            "delta_success_rate": float(cand_overall.get("success_rate", 0.0))
            - float(base_overall.get("success_rate", 0.0)),
            "baseline_mean_completion": float(base_overall.get("mean_completion", 0.0)),
            "candidate_mean_completion": float(cand_overall.get("mean_completion", 0.0)),
            "delta_mean_completion": float(cand_overall.get("mean_completion", 0.0))
            - float(base_overall.get("mean_completion", 0.0)),
        },
    }


def format_evaluation_table(result: dict[str, Any]) -> str:
    """Human-readable CLI summary."""

    lines = ["level                     success  completion  return   episodes"]
    per_level = result.get("aggregate", {}).get("per_level", {})
    for level_id, data in per_level.items():
        lines.append(
            f"{level_id:<25} {data['success_rate'] * 100:6.1f}%  "
            f"{data['mean_completion'] * 100:9.1f}%  {data['mean_return']:7.2f}  "
            f"{data['episodes']:8d}"
        )
    overall = result.get("aggregate", {}).get("overall", {})
    lines.append(
        f"{'OVERALL':<25} {overall.get('success_rate', 0.0) * 100:6.1f}%  "
        f"{overall.get('mean_completion', 0.0) * 100:9.1f}%  "
        f"{overall.get('mean_return', 0.0):7.2f}  {overall.get('episodes', 0):8d}"
    )
    return "\n".join(lines)


def summarise_terminations(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = str(record.get("termination_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def env_config_from_checkpoint(extra: dict[str, Any], level_id: str) -> EnvConfig:
    """Rebuild the exact simulation dynamics recorded with a checkpoint.

    A checkpoint stores the training ``TrainConfig``, so evaluation, replay and the
    research-assistant probe can reproduce the same action range, control rate, actuation
    mode and reward scaling the policy was trained with.  Without this, a policy trained at
    20 Hz with ``action_scale=0.4`` would be evaluated at a different control rate and fall
    over for reasons that have nothing to do with its skill.
    """

    from .env import RewardConfig

    config = extra.get("config") or {}
    reward = RewardConfig.from_dict(config.get("reward", {}) or {})
    return EnvConfig(
        level_id=level_id,
        control_decimation=int(config.get("control_decimation", 5)),
        physics_timestep=float(config.get("physics_timestep", 0.002)),
        action_scale=float(config.get("action_scale", 0.25)),
        action_smoothing=float(config.get("action_smoothing", 0.6)),
        actuation=config.get("actuation", "position"),
        max_episode_seconds=config.get("max_episode_seconds"),
        reward=reward,
        # Use what the run actually trained with.  Callers that want a fixed initial state
        # (replays, matched-seed comparisons) pass ``randomize_reset`` explicitly.
        randomize_reset=bool(config.get("randomize_reset", True)),
    )


__all__ = [
    "DEFAULT_EVAL_SEEDS",
    "EvalConfig",
    "aggregate_records",
    "compare_aggregates",
    "evaluate_episode_set",
    "env_config_from_checkpoint",
    "evaluate_policy",
    "format_evaluation_table",
    "expand_eval_seeds",
    "rollout_episode",
    "summarise_terminations",
]
