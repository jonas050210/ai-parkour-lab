"""The training loop: PPO + parallel environments + adaptive levels + persistence.

One training "run" owns a fresh, immutable run directory.  Each iteration collects a fixed
number of transitions per environment, updates the policy with PPO, logs the iteration,
and periodically evaluates deterministically, checkpoints and records replays.  The
trainer never touches held-out level data unless the caller explicitly opts in.

Manual level selection and adaptive all-level training use the same loop; only the
sampler differs (``single``, ``uniform`` or ``adaptive``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..env import EnvConfig, RewardConfig
from ..evaluation import (
    EvalConfig,
    compare_aggregates,
    evaluate_policy,
    rollout_episode,
)
from ..levels import LevelCatalog
from ..policies import ActorCritic
from ..ppo import PPOConfig, Rollout, compute_gae, linear_schedule, ppo_update
from ..storage import DEFAULT_RUNS_DIR, RunStore, _jsonable
from .curriculum import CurriculumConfig, LevelSampler, make_sampler
from .vector import make_vec_env

TRAINING_COLUMNS = (
    "iteration",
    "total_steps",
    "wall_seconds",
    "steps_per_second",
    "episodes",
    "episode_return",
    "episode_length",
    "success_rate",
    "mean_completion",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "learning_rate",
    "eval_success_rate",
    "eval_mean_completion",
    "best_eval_success",
)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Complete, serialisable configuration of a training run."""

    level_ids: tuple[str, ...] = ()
    curriculum: str = "adaptive"
    seed: int = 0
    total_steps: int = 1_000_000
    rollout_steps: int = 256
    num_envs: int = 4
    workers: int = 1
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    reward_start: RewardConfig | None = None
    reward_ramp_fraction: float = 0.0
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)
    action_scale: float = 0.25
    action_smoothing: float = 0.6
    actuation: str = "position"
    physics_timestep: float = 0.002
    control_decimation: int = 5
    eval_interval: int = 20
    eval_levels: tuple[str, ...] = ()
    eval_seeds: tuple[int, ...] = (11, 23)
    eval_episodes_per_level: int = 1
    checkpoint_interval: int = 20
    keep_checkpoints: int = 6
    max_replays: int = 12
    record_replay_every_eval: int = 2
    max_minutes: float | None = None
    target_success: float | None = None
    reward_scale: float = 0.25
    max_episode_seconds: float | None = None
    stagnation_seconds: float = 0.0
    randomize_reset: bool = True
    normalize_observations: bool = True
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    initial_log_std: float = -1.0
    notes: str = ""
    name: str = ""

    # ----------------------------------------------------------------------------------
    def env_config(self, level_id: str) -> EnvConfig:
        return EnvConfig(
            level_id=level_id,
            control_decimation=self.control_decimation,
            physics_timestep=self.physics_timestep,
            action_scale=self.action_scale,
            action_smoothing=self.action_smoothing,
            actuation=self.actuation,  # type: ignore[arg-type]
            reward=self.reward,
            randomize_reset=self.randomize_reset,
            max_episode_seconds=self.max_episode_seconds,
            stagnation_seconds=self.stagnation_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "notes": self.notes,
            "level_ids": list(self.level_ids),
            "curriculum": self.curriculum,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "num_envs": self.num_envs,
            "workers": self.workers,
            "ppo": self.ppo.to_dict(),
            "reward": self.reward.to_dict(),
            "curriculum_config": self.curriculum_config.to_dict(),
            "action_scale": self.action_scale,
            "action_smoothing": self.action_smoothing,
            "actuation": self.actuation,
            "physics_timestep": self.physics_timestep,
            "control_decimation": self.control_decimation,
            "eval_interval": self.eval_interval,
            "eval_levels": list(self.eval_levels),
            "eval_seeds": list(self.eval_seeds),
            "eval_episodes_per_level": self.eval_episodes_per_level,
            "checkpoint_interval": self.checkpoint_interval,
            "keep_checkpoints": self.keep_checkpoints,
            "max_replays": self.max_replays,
            "record_replay_every_eval": self.record_replay_every_eval,
            "max_minutes": self.max_minutes,
            "target_success": self.target_success,
            "reward_start": None if self.reward_start is None else self.reward_start.to_dict(),
            "reward_ramp_fraction": self.reward_ramp_fraction,
            "reward_scale": self.reward_scale,
            "max_episode_seconds": self.max_episode_seconds,
            "stagnation_seconds": self.stagnation_seconds,
            "randomize_reset": self.randomize_reset,
            "normalize_observations": self.normalize_observations,
            "hidden_dim": self.hidden_dim,
            "initial_log_std": self.initial_log_std,
            "learning_rate": self.learning_rate,
            "control_dt": self.control_decimation * self.physics_timestep,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(
            name=str(data.get("name", "")),
            notes=str(data.get("notes", "")),
            level_ids=tuple(data.get("level_ids", ())),
            curriculum=str(data.get("curriculum", "adaptive")),
            seed=int(data.get("seed", 0)),
            total_steps=int(data.get("total_steps", 1_000_000)),
            rollout_steps=int(data.get("rollout_steps", 256)),
            num_envs=int(data.get("num_envs", 4)),
            workers=int(data.get("workers", 1)),
            ppo=PPOConfig.from_dict(data.get("ppo", {})),
            reward=RewardConfig.from_dict(data.get("reward", {})),
            curriculum_config=CurriculumConfig.from_dict(data.get("curriculum_config", {})),
            action_scale=float(data.get("action_scale", 0.25)),
            action_smoothing=float(data.get("action_smoothing", 0.6)),
            actuation=str(data.get("actuation", "position")),
            physics_timestep=float(data.get("physics_timestep", 0.002)),
            control_decimation=int(data.get("control_decimation", 5)),
            eval_interval=int(data.get("eval_interval", 20)),
            eval_levels=tuple(data.get("eval_levels", ())),
            eval_seeds=tuple(int(seed) for seed in data.get("eval_seeds", (11, 23))),
            eval_episodes_per_level=int(data.get("eval_episodes_per_level", 1)),
            checkpoint_interval=int(data.get("checkpoint_interval", 20)),
            keep_checkpoints=int(data.get("keep_checkpoints", 6)),
            max_replays=int(data.get("max_replays", 12)),
            record_replay_every_eval=int(data.get("record_replay_every_eval", 2)),
            max_minutes=(None if data.get("max_minutes") is None else float(data["max_minutes"])),
            target_success=(
                None if data.get("target_success") is None else float(data["target_success"])
            ),
            reward_start=(
                None
                if data.get("reward_start") is None
                else RewardConfig.from_dict(data["reward_start"])
            ),
            reward_ramp_fraction=float(data.get("reward_ramp_fraction", 0.0)),
            reward_scale=float(data.get("reward_scale", 0.25)),
            max_episode_seconds=(
                None
                if data.get("max_episode_seconds") is None
                else float(data["max_episode_seconds"])
            ),
            stagnation_seconds=float(data.get("stagnation_seconds", 0.0)),
            randomize_reset=bool(data.get("randomize_reset", True)),
            normalize_observations=bool(data.get("normalize_observations", True)),
            hidden_dim=int(data.get("hidden_dim", 128)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            initial_log_std=float(data.get("initial_log_std", -1.0)),
        )


class TrainingRunner:
    """Owns one training run, its checkpoints and its metrics."""

    def __init__(
        self,
        config: TrainConfig,
        catalog: LevelCatalog,
        store: RunStore,
        *,
        logger: Callable[[str], None] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        resume_from: Path | str | None = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.store = store
        self.logger = logger or (lambda message: print(message, flush=True))
        self.progress = progress
        self.rng = np.random.default_rng(config.seed)
        observation_dim, action_dim = _probe_dimensions(catalog, config)
        self.model = ActorCritic(
            observation_dim,
            action_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
            learning_rate=config.learning_rate,
            initial_log_std=config.initial_log_std,
            normalize_observations=config.normalize_observations,
        )
        self.sampler: LevelSampler | None = None
        self.iteration = 0
        self.total_steps = 0
        self.episodes = 0
        self.best_eval_success = -1.0
        self.best_eval_completion = -1.0
        self.best_score = -math.inf
        self.history: list[dict[str, Any]] = []
        self.episode_stats: list[dict[str, Any]] = []
        self.vec = None
        self._start_time = time.monotonic()
        self._eval_counter = 0
        self._checkpoint_files: list[Path] = []
        self._last_eval: dict[str, Any] | None = None
        self._stopped = False
        self._resume_path = Path(resume_from) if resume_from is not None else None
        self._active_reward = config.reward
        self._reward_ramp_announced = False

    # ----------------------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------------------
    def _resolve_levels(self) -> tuple[str, ...]:
        if self.config.level_ids:
            ids = tuple(self.config.level_ids)
        else:
            ids = tuple(level.id for level in self.catalog.train_levels)
        for level_id in ids:
            self.catalog[level_id]
        return ids

    def _resolve_eval_levels(self, train_levels: Sequence[str]) -> tuple[str, ...]:
        if self.config.eval_levels:
            return tuple(self.config.eval_levels)
        if len(train_levels) <= 4:
            return tuple(train_levels)
        # Evenly sample the training levels so the evaluation covers easy and hard courses.
        indices = np.linspace(0, len(train_levels) - 1, 4).round().astype(int)
        return tuple(sorted({train_levels[int(index)] for index in indices}))

    def setup(self) -> None:
        config = self.config
        level_ids = self._resolve_levels()
        self.eval_levels = self._resolve_eval_levels(level_ids)
        self.sampler = make_sampler(
            [self.catalog[level_id] for level_id in level_ids],
            mode=config.curriculum,
            seed=config.seed,
            config=config.curriculum_config,
        )
        assigned = self.sampler.sample(config.num_envs)
        self.assigned_levels = assigned
        self.vec = make_vec_env(
            self.catalog,
            assigned,
            env_config=config.env_config(assigned[0]),
            seeds=[config.seed + 1000 * index for index in range(config.num_envs)],
            workers=config.workers,
        )
        if self.model.observation_dim != self.vec.observation_dim:
            raise ValueError(
                f"policy expects {self.model.observation_dim} observations but the env produces "
                f"{self.vec.observation_dim}"
            )
        if self.model.action_dim != self.vec.action_dim:
            raise ValueError(
                f"policy emits {self.model.action_dim} actions but the env expects "
                f"{self.vec.action_dim}"
            )
        self._apply_reward_schedule()
        if self._resume_path is not None:
            self.load_checkpoint(self._resume_path)
        self.observation = self.vec.reset()
        self.store.update_summary(
            status="running",
            pid=os.getpid(),
            config=config.to_dict(),
            levels=list(level_ids),
            eval_levels=list(self.eval_levels),
            total_steps=config.total_steps,
        )
        self.logger(
            f"run {self.store.paths.run_id}: {len(level_ids)} level(s), "
            f"{config.num_envs} env(s), {config.workers} worker(s), "
            f"{config.total_steps:,} steps, curriculum={config.curriculum}"
        )

    # ----------------------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self.setup()
        config = self.config
        assert self.vec is not None and self.sampler is not None
        deadline = (
            time.monotonic() + config.max_minutes * 60.0
            if config.max_minutes is not None
            else None
        )
        try:
            while self.total_steps < config.total_steps:
                if deadline is not None and time.monotonic() > deadline:
                    self.logger(f"stopping: max_minutes={config.max_minutes} reached")
                    break
                iteration_start = time.monotonic()
                self._apply_reward_schedule()
                batch, metrics = self._collect_rollout()
                fraction = min(1.0, self.total_steps / max(1, config.total_steps))
                learning_rate = linear_schedule(
                    config.learning_rate,
                    config.ppo.learning_rate_final_fraction,
                    fraction,
                )
                stats = ppo_update(
                    self.model, batch, config.ppo, self.rng, learning_rate=learning_rate
                )
                self.iteration += 1
                wall = time.monotonic() - self._start_time
                row: dict[str, Any] = {
                    "iteration": self.iteration,
                    "total_steps": self.total_steps,
                    "wall_seconds": round(wall, 3),
                    "steps_per_second": round(
                        metrics["transitions"] / max(1e-9, time.monotonic() - iteration_start), 1
                    ),
                    "episodes": metrics["episodes"],
                    "episode_return": round(metrics["episode_return"], 4),
                    "episode_length": round(metrics["episode_length"], 2),
                    "success_rate": round(metrics["success_rate"], 4),
                    "mean_completion": round(metrics["mean_completion"], 4),
                    "policy_loss": round(stats.policy_loss, 6),
                    "value_loss": round(stats.value_loss, 6),
                    "entropy": round(stats.entropy, 5),
                    "approx_kl": round(stats.approx_kl, 6),
                    "clip_fraction": round(stats.clip_fraction, 5),
                    "explained_variance": round(stats.explained_variance, 5),
                    "learning_rate": learning_rate,
                    "eval_success_rate": "",
                    "eval_mean_completion": "",
                    "best_eval_success": round(self.best_eval_success, 4)
                    if self.best_eval_success >= 0
                    else "",
                }

                if self.iteration % config.eval_interval == 0 or self.total_steps >= config.total_steps:
                    evaluation = self.evaluate()
                    row["eval_success_rate"] = round(
                        evaluation["aggregate"]["overall"]["success_rate"], 4
                    )
                    row["eval_mean_completion"] = round(
                        evaluation["aggregate"]["overall"]["mean_completion"], 4
                    )
                    row["best_eval_success"] = round(self.best_eval_success, 4)

                self.store.append_training(row)
                self.history.append(row)
                self.logger(self._format_row(row))
                self._emit_progress(row, evaluation=row.get("eval_success_rate", ""))

                if self.iteration % config.checkpoint_interval == 0:
                    self.save_checkpoint("latest")
                    self.save_checkpoint(f"iter_{self.iteration:05d}")
                if config.target_success is not None and self.best_eval_success >= config.target_success:
                    self.logger(
                        f"target success {config.target_success:.2f} reached at "
                        f"{self.total_steps:,} steps"
                    )
                    break
        except KeyboardInterrupt:
            self.logger("interrupted: saving checkpoint and summary")
            self._stopped = True
        finally:
            self.save_checkpoint("latest")
            summary = self._final_summary()
            self.store.update_summary(**summary)
            self.vec.close()
        return self._final_summary()

    def _apply_reward_schedule(self) -> None:
        """Annealed reward shaping.

        With ``reward_ramp_fraction > 0`` the weights blend from ``reward_start`` to
        ``reward`` over the first ``reward_ramp_fraction`` of the run.  Starting from
        posture-only shaping stops the policy from discovering that falling forward towards
        the goal pays before it has learned to stand up.
        """

        config = self.config
        start = config.reward_start
        if start is None or config.reward_ramp_fraction <= 0.0:
            return
        assert self.vec is not None
        span = max(1.0, config.reward_ramp_fraction * config.total_steps)
        alpha = min(1.0, self.total_steps / span)
        reward = start.blend(config.reward, alpha)
        if reward != self._active_reward:
            self._active_reward = reward
            self.vec.set_reward(reward)
            if alpha >= 1.0 and not self._reward_ramp_announced:
                self._reward_ramp_announced = True
                self.logger(
                    f"reward ramp complete at {self.total_steps:,} steps; "
                    "using the final reward weights"
                )

    # ----------------------------------------------------------------------------------
    # Rollout collection
    # ----------------------------------------------------------------------------------
    def _collect_rollout(self) -> tuple[Rollout, dict[str, Any]]:
        config = self.config
        assert self.vec is not None and self.sampler is not None
        steps = config.rollout_steps
        num_envs = config.num_envs
        observation_dim = self.vec.observation_dim
        action_dim = self.vec.action_dim
        observations = np.zeros((steps, num_envs, observation_dim), dtype=np.float64)
        actions = np.zeros((steps, num_envs, action_dim), dtype=np.float64)
        log_probs = np.zeros((steps, num_envs), dtype=np.float64)
        values = np.zeros((steps, num_envs), dtype=np.float64)
        rewards = np.zeros((steps, num_envs), dtype=np.float64)
        dones = np.zeros((steps, num_envs), dtype=np.float64)
        episode_returns: list[float] = []
        episode_lengths: list[float] = []
        successes = 0
        completions: list[float] = []

        for step in range(steps):
            actions_t, log_prob_t, value_t = self.model.sample(self.observation, self.rng)
            (
                next_observation,
                rewards_t,
                terminated,
                truncated,
                infos,
            ) = self.vec.step(actions_t)
            observations[step] = self.observation
            actions[step] = actions_t
            log_probs[step] = log_prob_t
            values[step] = value_t
            rewards[step] = rewards_t * config.reward_scale
            dones[step] = np.logical_or(terminated, truncated)
            self.total_steps += num_envs
            levels_changed = False
            for index, info in enumerate(infos):
                episode = info.get("episode")
                if not episode:
                    continue
                self.episodes += 1
                episode["iteration"] = self.iteration + 1
                episode["total_steps"] = self.total_steps
                self.store.append_episode(episode)
                episode_returns.append(float(episode.get("episode_return", 0.0)))
                episode_lengths.append(float(episode.get("sim_steps", 0)))
                successes += int(bool(episode.get("success", False)))
                completions.append(float(episode.get("completion_fraction", 0.0)))
                self.sampler.update(
                    episode["level_id"],
                    success=bool(episode.get("success", False)),
                    progress=float(episode.get("completion_fraction", 0.0)),
                    episode_return=float(episode.get("episode_return", 0.0)),
                    step=self.total_steps,
                )
                if config.curriculum != "single":
                    new_level = self.sampler.sample(1)[0]
                    if new_level != self.assigned_levels[index]:
                        self.assigned_levels[index] = new_level
                        levels_changed = True
            if levels_changed:
                self.vec.set_levels(self.assigned_levels)
            self.observation = next_observation

        last_values = self.model.value(self.observation)
        flat_observations = observations.reshape(steps * num_envs, observation_dim)
        flat_actions = actions.reshape(steps * num_envs, action_dim)
        flat_log_probs = log_probs.reshape(-1)
        flat_values = values.reshape(-1)
        flat_rewards = rewards.reshape(-1)
        advantages = np.zeros_like(values)
        for env_index in range(num_envs):
            env_advantages, _ = compute_gae(
                rewards[:, env_index],
                values[:, env_index],
                dones[:, env_index],
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                last_value=float(last_values[env_index]),
            )
            advantages[:, env_index] = env_advantages
        returns = advantages + values
        self.model.obs_norm.update(flat_observations)
        batch = Rollout(
            observations=flat_observations,
            actions=flat_actions,
            log_probs=flat_log_probs,
            values=flat_values,
            rewards=flat_rewards,
            dones=dones.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )
        metrics = {
            "transitions": steps * num_envs,
            "episodes": len(episode_returns),
            "episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "success_rate": successes / len(episode_returns) if episode_returns else 0.0,
            "mean_completion": float(np.mean(completions)) if completions else 0.0,
        }
        return batch, metrics

    # ----------------------------------------------------------------------------------
    # Evaluation, checkpoints, replays
    # ----------------------------------------------------------------------------------
    def evaluate(self, *, record_replay: bool | None = None) -> dict[str, Any]:
        config = self.config
        self._eval_counter += 1
        result = evaluate_policy(
            self.model,
            self.catalog,
            self.eval_levels,
            config=EvalConfig(
                seeds=config.eval_seeds[: config.eval_episodes_per_level],
                deterministic=True,
            ),
            env_config=config.env_config(self.eval_levels[0]),
        )
        overall = result["aggregate"]["overall"]
        result["iteration"] = self.iteration
        result["total_steps"] = self.total_steps
        self._last_eval = result
        self.store.write_json(f"evaluation/iter_{self.iteration:05d}.json", result)
        self.store.append_evaluation(
            {
                "iteration": self.iteration,
                "total_steps": self.total_steps,
                "episodes": overall["episodes"],
                "success_rate": round(overall["success_rate"], 4),
                "mean_completion": round(overall["mean_completion"], 4),
                "mean_return": round(overall["mean_return"], 4),
                "levels": ";".join(self.eval_levels),
            }
        )
        score = overall["success_rate"] + 0.25 * overall["mean_completion"]
        if score > self.best_score:
            self.best_score = score
            self.best_eval_success = overall["success_rate"]
            self.best_eval_completion = overall["mean_completion"]
            self.save_checkpoint("best")
        should_record = (
            config.record_replay_every_eval > 0
            and self._eval_counter % config.record_replay_every_eval == 0
        ) if record_replay is None else record_replay
        if should_record:
            self._record_replay()
        self.logger(
            f"  eval @ {self.total_steps:,}: success={overall['success_rate'] * 100:.1f}% "
            f"completion={overall['mean_completion'] * 100:.1f}% "
            f"(best {self.best_eval_success * 100:.1f}%)"
        )
        return result

    def _record_replay(self) -> None:
        """Record one deterministic episode for the 3D viewer (bounded file count)."""

        config = self.config
        level_id = self.eval_levels[-1]
        seed = int(config.eval_seeds[0])
        try:
            record = rollout_episode(
                self.model,
                self.catalog[level_id],
                seed=seed,
                env_config=config.env_config(level_id),
                deterministic=True,
                record_frames=True,
                frame_stride=2,  # 25 fps at the 50 Hz default control rate
            )
        except Exception as exc:  # pragma: no cover - replay is best effort
            self.logger(f"  replay recording failed: {type(exc).__name__}: {exc}")
            return
        payload = {
            "run_id": self.store.paths.run_id,
            "level_id": level_id,
            "level_title": self.catalog[level_id].title,
            "seed": seed,
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "success": record["success"],
            "termination_reason": record["termination_reason"],
            "completion_fraction": record["completion_fraction"],
            "episode_return": record["episode_return"],
            "frames": record["frames"],
            "level": self.catalog[level_id].to_dict(),
        }
        self.store.save_replay(
            f"iter{self.iteration:05d}-{level_id}-seed{seed}", payload
        )
        replays = sorted(self.store.paths.replay_dir.glob("*.json.gz"))
        while len(replays) > config.max_replays:
            replays.pop(0).unlink(missing_ok=True)

    def save_checkpoint(self, name: str) -> Path:
        extra = {
            "run_id": self.store.paths.run_id,
            "level_ids": list(self.config.level_ids),
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "best_eval_success": self.best_eval_success,
            "best_eval_completion": self.best_eval_completion,
            "best_score": self.best_score,
            "config": self.config.to_dict(),
            "sampler": self.sampler.state_dict() if self.sampler is not None else None,
            "assigned_levels": getattr(self, "assigned_levels", []),
            "rng_state": _jsonable(self.rng.bit_generator.state),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "episode_stats": [float(value) for value in self.episode_stats[-50:]],
        }
        path = self.store.paths.checkpoint_dir / f"{name}.npz"
        self.model.save(path, extra=extra)
        if name == "latest":
            self._checkpoint_files.append(path)
            self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        keep = self.config.keep_checkpoints
        if keep <= 0:
            return
        files = sorted(
            self.store.paths.checkpoint_dir.glob("iter_*.npz"),
            key=lambda path: path.stat().st_mtime,
        )
        while len(files) > keep:
            files.pop(0).unlink(missing_ok=True)

    def load_checkpoint(self, path: Path | str) -> None:
        model, extra = ActorCritic.load(path, seed=self.config.seed)
        if model.observation_dim != self.model.observation_dim:
            raise ValueError(
                f"checkpoint observation dim {model.observation_dim} != "
                f"{self.model.observation_dim}"
            )
        self.model.load_state_dict(model.state_dict())
        self.iteration = int(extra.get("iteration", 0))
        self.total_steps = int(extra.get("total_steps", 0))
        self.episodes = int(extra.get("episodes", 0))
        self.best_eval_success = float(extra.get("best_eval_success", -1.0))
        self.best_eval_completion = float(extra.get("best_eval_completion", -1.0))
        self.best_score = float(extra.get("best_score", -math.inf))
        if "rng_state" in extra:
            self.rng.bit_generator.state = extra["rng_state"]
        if self.sampler is not None and extra.get("sampler"):
            self.sampler.load_state_dict(extra["sampler"])
        assigned = extra.get("assigned_levels") or []
        if assigned and len(assigned) == self.config.num_envs and self.vec is not None:
            self.assigned_levels = list(assigned)
            self.vec.set_levels(self.assigned_levels)
        self.logger(
            f"resumed from {Path(path).name} at iteration {self.iteration}, "
            f"{self.total_steps:,} steps"
        )
        self.store.update_summary(status="resumed", resumed_from=str(path))

    # ----------------------------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------------------------
    def _format_row(self, row: dict[str, Any]) -> str:
        eval_part = ""
        if row.get("eval_success_rate") not in ("", None):
            eval_part = f" | eval success {float(row['eval_success_rate']) * 100:5.1f}%"
        return (
            f"iter {row['iteration']:4d} | {row['total_steps']:>8,} steps | "
            f"{row['steps_per_second']:7.0f} steps/s | return {row['episode_return']:8.2f} | "
            f"episodes {row['episodes']:3d} | train success {row['success_rate'] * 100:5.1f}%"
            f"{eval_part}"
        )

    def _emit_progress(self, row: dict[str, Any], **extra: Any) -> None:
        if self.progress is None:
            return
        payload = {"type": "iteration", **row, **extra}
        self.progress(payload)

    def _final_summary(self) -> dict[str, Any]:
        summary = self.store.read_summary()
        status = "interrupted" if self._stopped else "completed"
        if self.total_steps < self.config.total_steps and not self._stopped:
            status = "stopped_early"
        overall = (
            self._last_eval["aggregate"]["overall"] if self._last_eval is not None else {}
        )
        sampler_stats = self.sampler.stats() if self.sampler is not None else {}
        data = {
            "status": status,
            "total_steps": self.total_steps,
            "iterations": self.iteration,
            "episodes": self.episodes,
            "best_eval_success": self.best_eval_success,
            "best_eval_completion": self.best_eval_completion,
            "last_eval": overall,
            "final_success_rate": overall.get("success_rate"),
            "final_mean_completion": overall.get("mean_completion"),
            "wall_seconds": round(time.monotonic() - self._start_time, 2),
            "curriculum": sampler_stats,
            "levels": list(getattr(self, "eval_levels", ())),
            "checkpoints": [row["name"] for row in self.store.checkpoints()],
            "previous": {
                key: summary.get(key)
                for key in ("created_utc", "resumed_from")
                if key in summary
            },
        }
        self.store.update_summary(**data)
        return data


def _probe_dimensions(catalog: LevelCatalog, config: TrainConfig) -> tuple[int, int]:
    """Observation/action dimensions of the environment configuration."""

    from ..env import ParkourEnv

    level = catalog[config.level_ids[0]] if config.level_ids else catalog.train_levels[0]
    env = ParkourEnv(level, config.env_config(level.id))
    try:
        return env.observation_dim, env.action_dim
    finally:
        env.close()


def train(
    config: TrainConfig | None,
    catalog: LevelCatalog,
    *,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    run_id: str | None = None,
    resume_from: Path | str | None = None,
    logger: Callable[[str], None] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Create (or reopen) a run directory and execute the training loop.

    With ``resume_from`` the checkpoint's run directory is reused, its saved configuration
    is restored, and the policy, optimiser, sampler and counters continue from the
    checkpoint rather than from scratch.
    """

    if resume_from is not None:
        checkpoint = Path(resume_from)
        run_root = checkpoint.parent.parent
        store = RunStore.load(run_root)
        resolved = TrainConfig.from_dict(store.read_config())
        if run_id is not None and run_id != store.paths.run_id:
            raise ValueError(
                f"run_id {run_id!r} does not match the checkpoint's run {store.paths.run_id!r}"
            )
    else:
        if config is None:
            raise ValueError("a training configuration is required for a new run")
        resolved = config
        store = RunStore.create(
            runs_dir,
            run_id=run_id,
            name=resolve_name(resolved),
            config=resolved.to_dict(),
        )
    runner = TrainingRunner(
        resolved,
        catalog,
        store,
        logger=logger,
        progress=progress,
        resume_from=resume_from,
    )
    summary = runner.run()
    summary["run_id"] = store.paths.run_id
    summary["run_dir"] = str(store.paths.root)
    return summary


def resolve_name(config: TrainConfig) -> str:
    if config.name:
        return config.name
    if len(config.level_ids) == 1:
        return config.level_ids[0].replace("level-", "l")
    return f"{config.curriculum}-ppo"


def compare_runs(
    store_a: RunStore, store_b: RunStore
) -> dict[str, Any]:
    """Compare the best evaluations of two runs on their shared levels."""

    def best(store: RunStore) -> dict[str, Any] | None:
        evaluations = store.evaluations()
        if not evaluations:
            return None
        return max(
            evaluations,
            key=lambda record: float(record.get("aggregate", {}).get("overall", {}).get("success_rate", 0.0)),
        )

    eval_a = best(store_a)
    eval_b = best(store_b)
    if eval_a is None or eval_b is None:
        return {
            "error": "both runs need at least one evaluation to be compared",
            "run_a": store_a.paths.run_id,
            "run_b": store_b.paths.run_id,
        }
    comparison = compare_aggregates(eval_a["aggregate"], eval_b["aggregate"])
    comparison.update(
        {
            "run_a": {
                "run_id": store_a.paths.run_id,
                "iteration": eval_a.get("iteration"),
                "total_steps": eval_a.get("total_steps"),
                "levels": eval_a.get("levels"),
            },
            "run_b": {
                "run_id": store_b.paths.run_id,
                "iteration": eval_b.get("iteration"),
                "total_steps": eval_b.get("total_steps"),
                "levels": eval_b.get("levels"),
            },
        }
    )
    return comparison


__all__ = [
    "TRAINING_COLUMNS",
    "TrainConfig",
    "TrainingRunner",
    "compare_runs",
    "resolve_name",
    "train",
]
