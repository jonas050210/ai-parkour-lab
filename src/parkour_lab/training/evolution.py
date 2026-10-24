"""Bounded evolutionary search over reward/PPO hyperparameters.

PPO is the primary learner; this module is a *secondary* optimiser that tunes the
hand-designed parts of the problem (reward weights, exploration noise, learning rate)
using the same training loop and evaluation protocol.  It is deliberately small:

* candidate vectors are sampled in a normalised cube and mapped through per-parameter
  bounds, so no candidate can produce a nonsensical configuration;
* fitness is measured by actually training a short PPO run and evaluating the resulting
  policy deterministically on fixed seeds --- there is no surrogate model;
* the search is seeded and every candidate, its config and its metrics are saved, so a
  result can be reproduced.

This is a cross-entropy-method (CEM) search with elitist refitting.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..env import RewardConfig
from ..levels import LevelCatalog
from ..ppo import PPOConfig
from ..storage import DEFAULT_RUNS_DIR, RunStore
from .runner import TrainConfig, train


@dataclass(frozen=True, slots=True)
class Parameter:
    """One tuned scalar with hard bounds."""

    name: str
    low: float
    high: float
    log_scale: bool = False
    description: str = ""

    def to_unit(self, value: float) -> float:
        value = float(np.clip(value, self.low, self.high))
        if self.log_scale:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, unit: float) -> float:
        unit = float(np.clip(unit, 0.0, 1.0))
        if self.log_scale:
            return math.exp(math.log(self.low) + unit * (math.log(self.high) - math.log(self.low)))
        return self.low + unit * (self.high - self.low)

    def clamp(self, value: float) -> float:
        return float(np.clip(float(value), self.low, self.high))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "log_scale": self.log_scale,
            "description": self.description,
        }


# Every knob the search is allowed to touch. Physics, level geometry, observations and the
# policy architecture are intentionally *not* evolvable.
PARAMETERS: tuple[Parameter, ...] = (
    Parameter("reward.progress_per_metre", 1.0, 12.0, description="forward progress reward"),
    Parameter("reward.alive_per_second", 0.0, 2.5, description="upright survival bonus"),
    Parameter("reward.upright_per_second", 0.0, 2.5, description="torso orientation bonus"),
    Parameter("reward.action_rate_penalty", 1e-4, 0.2, True, "action smoothness penalty"),
    Parameter("reward.fall_penalty", 1.0, 25.0, description="terminal fall penalty"),
    Parameter("ppo.learning_rate", 1e-4, 1.5e-3, True, "Adam learning rate"),
    Parameter("ppo.entropy_coefficient", 1e-5, 2e-2, True, "exploration bonus"),
    Parameter("action_scale", 0.08, 0.6, description="normalised action range"),
)
PARAMETER_INDEX = {parameter.name: parameter for parameter in PARAMETERS}


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    generations: int = 3
    population: int = 6
    elite_fraction: float = 0.34
    train_steps: int = 40_000
    num_envs: int = 4
    workers: int = 1
    seed: int = 0
    eval_seeds: tuple[int, ...] = (11, 23)
    initial_std: float = 0.25
    minimum_std: float = 0.03
    success_weight: float = 1.0
    completion_weight: float = 0.5
    fall_weight: float = 0.15
    eval_levels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionConfig":
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)


def default_vector() -> np.ndarray:
    """Unit-space centre of the search cube (the default configuration)."""

    defaults = {
        "reward.progress_per_metre": RewardConfig().progress_per_metre,
        "reward.alive_per_second": RewardConfig().alive_per_second,
        "reward.upright_per_second": RewardConfig().upright_per_second,
        "reward.action_rate_penalty": RewardConfig().action_rate_penalty,
        "reward.fall_penalty": RewardConfig().fall_penalty,
        "ppo.learning_rate": PPOConfig().learning_rate,
        "ppo.entropy_coefficient": PPOConfig().entropy_coefficient,
        "action_scale": 0.25,
    }
    return np.array(
        [PARAMETER_INDEX[name].to_unit(value) for name, value in defaults.items()],
        dtype=np.float64,
    )


def vector_to_overrides(vector: Sequence[float]) -> dict[str, float]:
    """Map a unit-space vector to bounded parameter values keyed by parameter name."""

    if len(vector) != len(PARAMETERS):
        raise ValueError(f"expected {len(PARAMETERS)} values, got {len(vector)}")
    return {
        parameter.name: parameter.from_unit(unit)
        for parameter, unit in zip(PARAMETERS, vector)
    }


def apply_overrides(base: TrainConfig, overrides: dict[str, float]) -> TrainConfig:
    """Return a copy of ``base`` with the given bounded overrides applied."""

    unknown = set(overrides) - set(PARAMETER_INDEX)
    if unknown:
        raise ValueError(f"unknown evolvable parameters: {sorted(unknown)}")
    bounded = {
        name: PARAMETER_INDEX[name].clamp(value) for name, value in overrides.items()
    }
    reward = replace(
        base.reward,
        **{
            key.split(".", 1)[1]: value
            for key, value in bounded.items()
            if key.startswith("reward.")
        },
    )
    ppo = replace(
        base.ppo,
        **{
            key.split(".", 1)[1]: value
            for key, value in bounded.items()
            if key.startswith("ppo.")
        },
    )
    return replace(
        base,
        reward=reward,
        ppo=ppo,
        action_scale=bounded.get("action_scale", base.action_scale),
    )


def _fitness(evaluation: dict[str, Any], config: EvolutionConfig) -> float:
    overall = evaluation.get("aggregate", {}).get("overall", {})
    success = float(overall.get("success_rate", 0.0))
    completion = float(overall.get("mean_completion", 0.0))
    fall_rate = 0.0
    for record in evaluation.get("records", []):
        if record.get("termination_reason") in ("fell_over", "collapsed", "void_fall"):
            fall_rate += 1.0
    fall_rate /= max(1, len(evaluation.get("records", [])))
    return (
        config.success_weight * success
        + config.completion_weight * completion
        - config.fall_weight * fall_rate
    )


@dataclass
class Candidate:
    generation: int
    index: int
    vector: np.ndarray
    overrides: dict[str, float]
    run_id: str = ""
    fitness: float = float("-inf")
    success_rate: float = 0.0
    completion: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "index": self.index,
            "run_id": self.run_id,
            "fitness": self.fitness,
            "success_rate": self.success_rate,
            "completion": self.completion,
            "overrides": self.overrides,
            "vector": [float(value) for value in self.vector],
            "error": self.error,
        }


def _sample(
    mean: np.ndarray, std: np.ndarray, rng: np.random.Generator, count: int
) -> list[np.ndarray]:
    samples = []
    for _ in range(count):
        vector = np.clip(rng.normal(mean, std), 0.0, 1.0)
        samples.append(vector)
    return samples


def evolve_reward_weights(
    catalog: LevelCatalog,
    level_ids: Sequence[str],
    *,
    config: EvolutionConfig | None = None,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    base_config: TrainConfig | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a bounded CEM search and return the best configuration found."""

    cfg = config or EvolutionConfig()
    if not level_ids:
        raise ValueError("at least one level is required")
    for level_id in level_ids:
        catalog[level_id]
    say = logger or (lambda message: print(message, flush=True))
    runs_path = Path(runs_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    evo_id = f"evolve-{stamp}"
    evo_dir = runs_path / evo_id
    evo_dir.mkdir(parents=True, exist_ok=True)
    history_path = evo_dir / "evolution.jsonl"

    base = base_config or TrainConfig(
        level_ids=tuple(level_ids),
        curriculum="single" if len(level_ids) == 1 else "adaptive",
        total_steps=cfg.train_steps,
        num_envs=cfg.num_envs,
        workers=cfg.workers,
        seed=cfg.seed,
        eval_interval=max(1, cfg.train_steps // max(1, cfg.num_envs * 256)),
        eval_levels=tuple(level_ids),
        eval_seeds=tuple(cfg.eval_seeds),
        checkpoint_interval=1_000_000,  # only the final checkpoint matters here
    )
    base = replace(base, total_steps=cfg.train_steps, level_ids=tuple(level_ids))

    mean = default_vector()
    std = np.full(len(PARAMETERS), float(cfg.initial_std))
    elite_count = max(1, int(round(cfg.population * cfg.elite_fraction)))
    rng = np.random.default_rng(cfg.seed)
    all_candidates: list[Candidate] = []
    best: Candidate | None = None

    say(
        f"evolution {evo_id}: {cfg.generations} generations x {cfg.population} candidates, "
        f"{cfg.train_steps:,} steps each on {len(level_ids)} level(s)"
    )
    for generation in range(cfg.generations):
        vectors = _sample(mean, std, rng, cfg.population)
        # Always re-evaluate the current mean (elitist, removes sampling noise drift).
        vectors[0] = mean.copy()
        generation_candidates: list[Candidate] = []
        for index, vector in enumerate(vectors):
            overrides = vector_to_overrides(vector)
            run_id = f"{evo_id}-g{generation}c{index}"
            candidate = Candidate(
                generation=generation,
                index=index,
                vector=vector,
                overrides=overrides,
                run_id=run_id,
            )
            say(
                f"  gen {generation} candidate {index}: "
                + ", ".join(f"{name}={value:.4g}" for name, value in overrides.items())
            )
            try:
                candidate_config = apply_overrides(base, overrides)
                candidate_config = replace(candidate_config, seed=cfg.seed + generation)
                train(
                    candidate_config,
                    catalog,
                    runs_dir=runs_path,
                    run_id=run_id,
                    logger=lambda message: None,
                )
                store = RunStore.load(runs_path / run_id)
                evaluations = store.evaluations()
                if not evaluations:
                    raise RuntimeError("candidate produced no evaluation record")
                evaluation = max(
                    evaluations,
                    key=lambda record: float(
                        record.get("aggregate", {}).get("overall", {}).get("success_rate", 0.0)
                    ),
                )
                candidate.fitness = _fitness(evaluation, cfg)
                overall = evaluation.get("aggregate", {}).get("overall", {})
                candidate.success_rate = float(overall.get("success_rate", 0.0))
                candidate.completion = float(overall.get("mean_completion", 0.0))
            except Exception as exc:  # a broken candidate must not stop the search
                candidate.error = f"{type(exc).__name__}: {exc}"
                say(f"    candidate failed: {candidate.error}")
            generation_candidates.append(candidate)
            all_candidates.append(candidate)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(candidate.to_dict()) + "\n")
            say(
                f"    fitness={candidate.fitness:.4f} success={candidate.success_rate * 100:.1f}% "
                f"completion={candidate.completion * 100:.1f}%"
            )

        scored = sorted(
            (candidate for candidate in generation_candidates if not candidate.error),
            key=lambda candidate: candidate.fitness,
            reverse=True,
        )
        if not scored:
            raise RuntimeError("every candidate in the generation failed; aborting evolution")
        elites = scored[:elite_count]
        elite_vectors = np.stack([candidate.vector for candidate in elites])
        mean = elite_vectors.mean(axis=0)
        std = np.maximum(elite_vectors.std(axis=0), cfg.minimum_std)
        generation_best = scored[0]
        if best is None or generation_best.fitness > best.fitness:
            best = generation_best
        say(
            f"  generation {generation} best: {generation_best.run_id} "
            f"fitness={generation_best.fitness:.4f}; sigma={np.round(std, 3).tolist()}"
        )

    assert best is not None
    best_config = apply_overrides(base, best.overrides)
    summary = {
        "evolution_id": evo_id,
        "generations": cfg.generations,
        "population": cfg.population,
        "train_steps_per_candidate": cfg.train_steps,
        "levels": list(level_ids),
        "best_run_id": best.run_id,
        "best_fitness": best.fitness,
        "best_overrides": best.overrides,
        "best_success_rate": best.success_rate,
        "best_completion": best.completion,
        "parameters": [parameter.to_dict() for parameter in PARAMETERS],
        "final_mean": vector_to_overrides(mean),
        "final_std": std.tolist(),
        "candidates": [candidate.to_dict() for candidate in all_candidates],
        "best_config": best_config.to_dict(),
        "note": (
            "Evolution only proposes bounded reward/PPO settings. It always trains PPO and "
            "evaluates the resulting policy; it never replaces the learning algorithm."
        ),
    }
    (evo_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "run_dir": str(evo_dir), "best": summary}


__all__ = [
    "Candidate",
    "EvolutionConfig",
    "PARAMETERS",
    "Parameter",
    "apply_overrides",
    "default_vector",
    "evolve_reward_weights",
    "vector_to_overrides",
]
