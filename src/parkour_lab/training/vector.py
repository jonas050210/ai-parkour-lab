"""Vectorised environment adapters for headless training.

Two interchangeable implementations are provided:

``SerialVecEnv``
    Runs N environments in the training process.  Simple, debuggable, ideal for tests and
    for single-worker runs.

``SubprocVecEnv``
    Runs one environment per worker process (``fork`` on Linux) and communicates over a
    pipe.  MuJoCo's Python bindings release the GIL, but separate processes still give the
    most predictable scaling on small CPU-only machines.  Every worker may switch levels
    between episodes, which is what adaptive all-level training needs.

Both expose the same interface: ``reset``, ``step``, ``set_levels`` and ``close``.  Levels
are assigned for the *next* reset so a mid-episode level change can never corrupt a
rollout.  Episode records are attached to the final ``info`` of each episode and include
the level, the seed, the termination reason and the reward/progress/safety metrics.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import replace
from typing import Any, Protocol, Sequence

import numpy as np

from ..env import EnvConfig, ParkourEnv, RewardConfig
from ..levels import LevelCatalog

DEFAULT_CACHE_SIZE = 24


class VectorEnv(Protocol):
    num_envs: int
    observation_dim: int
    action_dim: int

    def reset(self, seeds: Sequence[int] | None = None) -> np.ndarray: ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]: ...

    def set_levels(self, levels: Sequence[str]) -> None: ...

    def set_reward(self, reward: RewardConfig) -> None: ...

    def close(self) -> None: ...


class _EnvPool:
    """Shared level->environment cache with the reset/step bookkeeping."""

    def __init__(
        self,
        catalog: LevelCatalog,
        env_config: EnvConfig,
        levels: Sequence[str],
        seeds: Sequence[int],
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.catalog = catalog
        self.env_config = env_config
        self.levels = list(levels)
        self.seeds = list(seeds)
        self.cache_size = cache_size
        # One environment instance per (slot, level): envs are stateful, so two slots that
        # happen to share a level must never share an environment object.
        self._envs: list[dict[str, ParkourEnv]] = [{} for _ in self.levels]
        self._episode_counts = [0 for _ in self.levels]
        self._needs_reset = [False for _ in self.levels]
        self._current: list[ParkourEnv] = []
        for index, level_id in enumerate(self.levels):
            self._current.append(self._make_env(level_id, index))

    def _make_env(self, level_id: str, slot: int) -> ParkourEnv:
        cache = self._envs[slot]
        env = cache.get(level_id)
        if env is None:
            while len(cache) >= self.cache_size:
                stale_level, stale_env = next(iter(cache.items()))
                cache.pop(stale_level).close()
                del stale_env
            env = ParkourEnv(
                self.catalog[level_id],
                replace(self.env_config, level_id=level_id, seed=self.seeds[slot]),
            )
            cache[level_id] = env
        return env

    @property
    def observation_dim(self) -> int:
        return self._current[0].observation_dim

    @property
    def action_dim(self) -> int:
        return self._current[0].action_dim

    def reset(self, seeds: Sequence[int] | None = None) -> np.ndarray:
        observations = []
        for slot, env in enumerate(self._current):
            seed = self.seeds[slot] if seeds is None else int(seeds[slot])
            observation, _ = env.reset(seed=seed)
            self._needs_reset[slot] = False
            observations.append(observation)
        return np.asarray(observations, dtype=np.float32)

    def set_levels(self, levels: Sequence[str]) -> None:
        """Assign levels for each environment's *next* episode."""

        if len(levels) != len(self.levels):
            raise ValueError("expected one level id per environment")
        self.levels = [str(level_id) for level_id in levels]

    def set_reward(self, reward: RewardConfig) -> None:
        """Apply new reward weights to every cached environment and to future ones."""

        self.env_config = replace(self.env_config, reward=reward)
        for cache in self._envs:
            for env in cache.values():
                env.set_reward(reward)

    def _episode_seed(self, slot: int) -> int:
        self._episode_counts[slot] += 1
        return int(self.seeds[slot] + 7919 * self._episode_counts[slot])

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.float64)
        observations = np.zeros(
            (len(self._current), self.observation_dim), dtype=np.float32
        )
        rewards = np.zeros(len(self._current), dtype=np.float32)
        terminated = np.zeros(len(self._current), dtype=bool)
        truncated = np.zeros(len(self._current), dtype=bool)
        infos: list[dict[str, Any]] = []
        for slot, env in enumerate(self._current):
            if self._needs_reset[slot]:
                env = self._make_env(self.levels[slot], slot)
                self._current[slot] = env
                self._needs_reset[slot] = False
                env.reset(seed=self._episode_seed(slot))
            result = env.step(actions[slot])
            rewards[slot] = result.reward
            terminated[slot] = result.terminated
            truncated[slot] = result.truncated
            info = dict(result.info)
            observations[slot] = result.observation
            if result.terminated or result.truncated:
                info["episode"] = self._episode_record(env, slot)
                # The reset (and any new level assignment) is applied at the start of the
                # next step call so that trainers can re-assign levels first.
                self._needs_reset[slot] = True
            infos.append(info)
        return observations, rewards, terminated, truncated, infos

    def _episode_record(self, env: ParkourEnv, slot: int) -> dict[str, Any]:
        summary = env.episode_summary()
        summary["level_id"] = self.levels[slot]
        summary["env_index"] = slot
        return summary

    def close(self) -> None:
        for cache in self._envs:
            for env in cache.values():
                env.close()
            cache.clear()
        self._envs = []
        self._current = []


class SerialVecEnv:
    """N environments inside the current process."""

    def __init__(
        self,
        catalog: LevelCatalog,
        levels: Sequence[str],
        *,
        env_config: EnvConfig | None = None,
        seeds: Sequence[int] | None = None,
    ) -> None:
        if not levels:
            raise ValueError("at least one level is required")
        self.catalog = catalog
        base_config = env_config if env_config is not None else EnvConfig(level_id=levels[0])
        seeds = list(seeds) if seeds is not None else list(range(len(levels)))
        if len(seeds) != len(levels):
            raise ValueError("one seed per environment is required")
        self._pool = _EnvPool(catalog, base_config, levels, seeds)
        self.num_envs = len(levels)

    @property
    def observation_dim(self) -> int:
        return self._pool.observation_dim

    @property
    def action_dim(self) -> int:
        return self._pool.action_dim

    @property
    def levels(self) -> list[str]:
        return list(self._pool.levels)

    def reset(self, seeds: Sequence[int] | None = None) -> np.ndarray:
        return self._pool.reset(seeds)

    def step(self, actions: np.ndarray):
        return self._pool.step(actions)

    def set_levels(self, levels: Sequence[str]) -> None:
        self._pool.set_levels(levels)

    def set_reward(self, reward: RewardConfig) -> None:
        self._pool.set_reward(reward)

    def close(self) -> None:
        self._pool.close()


def _worker_main(
    connection: Any,
    catalog_dir: str,
    levels: list[str],
    env_config: EnvConfig,
    seeds: list[int],
    cache_size: int,
) -> None:  # pragma: no cover - exercised through SubprocVecEnv tests
    try:
        catalog = LevelCatalog.load(catalog_dir)
        pool = _EnvPool(catalog, env_config, levels, seeds, cache_size=cache_size)
        while True:
            message = connection.recv()
            command = message.get("cmd")
            if command == "close":
                pool.close()
                connection.send({"ok": True})
                break
            if command == "reset":
                observations = pool.reset(message.get("seeds"))
                connection.send({"obs": observations})
            elif command == "step":
                observations, rewards, terminated, truncated, infos = pool.step(
                    message["actions"]
                )
                connection.send(
                    {
                        "obs": observations,
                        "rewards": rewards,
                        "terminated": terminated,
                        "truncated": truncated,
                        "infos": infos,
                    }
                )
            elif command == "set_levels":
                pool.set_levels(message["levels"])
                connection.send({"ok": True, "levels": pool.levels})
            elif command == "set_reward":
                pool.set_reward(RewardConfig.from_dict(message["reward"]))
                connection.send({"ok": True})
            elif command == "ping":
                connection.send({"ok": True, "pid": os.getpid()})
            else:  # pragma: no cover - defensive
                connection.send({"error": f"unknown command {command!r}"})
    except Exception as exc:  # pragma: no cover - worker crash path
        try:
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            connection.close()
        raise
    finally:
        connection.close()


class SubprocVecEnv:
    """One environment per worker process, communicating over pipes."""

    def __init__(
        self,
        catalog: LevelCatalog,
        levels: Sequence[str],
        *,
        env_config: EnvConfig | None = None,
        seeds: Sequence[int] | None = None,
        workers: int | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        levels_dir: str | None = None,
    ) -> None:
        if not levels:
            raise ValueError("at least one level is required")
        self.catalog = catalog
        self.num_envs = len(levels)
        workers = workers or min(len(levels), max(1, (os.cpu_count() or 1)))
        self.workers = max(1, min(int(workers), self.num_envs))
        base_config = env_config if env_config is not None else EnvConfig(level_id=levels[0])
        seeds = list(seeds) if seeds is not None else list(range(self.num_envs))
        self._levels = list(levels)
        self._context = mp.get_context("fork")
        self._connections: list[Any] = []
        self._processes: list[Any] = []
        level_chunks = np.array_split(np.arange(self.num_envs), self.workers)
        self._chunks = [chunk for chunk in level_chunks if len(chunk) > 0]
        self._observation_dim = self._probe_dimensions(env_config, levels[0])
        self._action_dim = self._probe_dimensions(env_config, levels[0], action=True)
        directory = levels_dir or str(self._resolve_levels_dir())
        for chunk in self._chunks:
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_worker_main,
                args=(
                    child,
                    directory,
                    [self._levels[index] for index in chunk],
                    base_config,
                    [seeds[index] for index in chunk],
                    cache_size,
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    # ----------------------------------------------------------------------------------
    def _probe_dimensions(
        self, env_config: EnvConfig | None, level_id: str, *, action: bool = False
    ) -> int:
        probe = ParkourEnv(self.catalog[level_id], env_config)
        value = probe.action_dim if action else probe.observation_dim
        probe.close()
        return int(value)

    def _resolve_levels_dir(self) -> str:
        for level in self.catalog:
            if level.source_path is not None:
                return str(level.source_path.parent)
        from ..levels import DEFAULT_LEVELS_DIR

        return str(DEFAULT_LEVELS_DIR)

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def levels(self) -> list[str]:
        return list(self._levels)

    def _broadcast(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        for connection in self._connections:
            connection.send(message)
        responses = [connection.recv() for connection in self._connections]
        for response in responses:
            if "error" in response:
                raise RuntimeError(f"vector env worker error: {response['error']}")
        return responses

    def reset(self, seeds: Sequence[int] | None = None) -> np.ndarray:
        if seeds is None:
            responses = self._broadcast({"cmd": "reset"})
        else:
            seeds = list(seeds)
            responses = []
            for connection, chunk in zip(self._connections, self._chunks):
                connection.send({"cmd": "reset", "seeds": [seeds[i] for i in chunk]})
            responses = [connection.recv() for connection in self._connections]
        return np.concatenate([response["obs"] for response in responses], axis=0)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64)
        for connection, chunk in zip(self._connections, self._chunks):
            connection.send({"cmd": "step", "actions": actions[chunk]})
        observations, rewards, terminated, truncated, infos = [], [], [], [], []
        for response in (connection.recv() for connection in self._connections):
            if "error" in response:
                raise RuntimeError(f"vector env worker error: {response['error']}")
            observations.append(response["obs"])
            rewards.append(response["rewards"])
            terminated.append(response["terminated"])
            truncated.append(response["truncated"])
            infos.extend(response["infos"])
        return (
            np.concatenate(observations, axis=0),
            np.concatenate(rewards, axis=0),
            np.concatenate(terminated, axis=0),
            np.concatenate(truncated, axis=0),
            infos,
        )

    def set_levels(self, levels: Sequence[str]) -> None:
        if len(levels) != self.num_envs:
            raise ValueError("expected one level id per environment")
        for connection, chunk in zip(self._connections, self._chunks):
            connection.send(
                {"cmd": "set_levels", "levels": [levels[index] for index in chunk]}
            )
        for connection in self._connections:
            response = connection.recv()
            if "error" in response:
                raise RuntimeError(f"vector env worker error: {response['error']}")
        self._levels = [str(level_id) for level_id in levels]

    def set_reward(self, reward: RewardConfig) -> None:
        for connection in self._connections:
            connection.send({"cmd": "set_reward", "reward": reward.to_dict()})
        for connection in self._connections:
            response = connection.recv()
            if "error" in response:
                raise RuntimeError(f"vector env worker error: {response['error']}")

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send({"cmd": "close"})
                connection.recv()
            except (EOFError, BrokenPipeError, OSError):  # pragma: no cover
                pass
            finally:
                connection.close()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():  # pragma: no cover
                process.terminate()
        self._connections = []
        self._processes = []


def make_vec_env(
    catalog: LevelCatalog,
    levels: Sequence[str],
    *,
    env_config: EnvConfig | None = None,
    seeds: Sequence[int] | None = None,
    workers: int = 1,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> VectorEnv:
    """Create a serial or multiprocess vector environment."""

    workers = max(1, int(workers))
    if workers == 1 or len(levels) == 1:
        return SerialVecEnv(catalog, levels, env_config=env_config, seeds=seeds)
    return SubprocVecEnv(
        catalog, levels, env_config=env_config, seeds=seeds, workers=workers, cache_size=cache_size
    )


def benchmark_simulation(
    catalog: LevelCatalog,
    *,
    levels: Sequence[str],
    num_envs: int,
    steps: int = 300,
    warmup: int = 20,
    env_config: EnvConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure raw environment throughput for a given worker layout."""

    level_ids = [levels[index % len(levels)] for index in range(num_envs)]
    env = make_vec_env(
        catalog,
        level_ids,
        env_config=env_config,
        seeds=[seed + index for index in range(num_envs)],
        workers=num_envs,
    )
    rng = np.random.default_rng(seed)
    try:
        env.reset()
        for _ in range(warmup):
            env.step(rng.uniform(-1.0, 1.0, (num_envs, env.action_dim)))
        start = time.perf_counter()
        transitions = 0
        for _ in range(steps):
            _obs, _rew, _term, _trunc, _info = env.step(
                rng.uniform(-1.0, 1.0, (num_envs, env.action_dim))
            )
            transitions += num_envs
        elapsed = time.perf_counter() - start
    finally:
        env.close()
    return {
        "num_envs": num_envs,
        "workers": num_envs,
        "steps": steps,
        "transitions": transitions,
        "elapsed_seconds": elapsed,
        "steps_per_second": transitions / elapsed if elapsed > 0 else 0.0,
        "transitions_per_second_per_env": transitions / elapsed / num_envs if elapsed else 0.0,
    }


__all__ = [
    "SerialVecEnv",
    "SubprocVecEnv",
    "VectorEnv",
    "benchmark_simulation",
    "make_vec_env",
]
