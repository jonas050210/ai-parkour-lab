"""Command-line interface for AI Parkour Lab.

Everything the project can do is reachable from here: inspect the model and levels,
run a random or trained policy headlessly, train (single level or adaptive all-level),
evaluate, diagnose and compare checkpoints, record/export replays, benchmark parallel
environments,
open the native MuJoCo viewer, and serve the research dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import __version__
from .diagnostics import diagnose_policy, format_diagnosis
from .env import ACTION_DIM, OBSERVATION_NAMES, EnvConfig, ParkourEnv
from .evaluation import (
    EvalConfig,
    env_config_from_checkpoint,
    evaluate_policy,
    format_evaluation_table,
    rollout_episode,
)
from .humanoid import default_humanoid, humanoid_xml
from .levels import DEFAULT_LEVELS_DIR, LevelCatalog, LevelError
from .policies import ActorCritic
from .storage import DEFAULT_RUNS_DIR, RunStore, list_runs
from .training.curriculum import CurriculumConfig
from .training.runner import TrainConfig, compare_runs, train
from .training.vector import benchmark_simulation

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _catalog(args: argparse.Namespace) -> LevelCatalog:
    return LevelCatalog.load(getattr(args, "levels_dir", None))


def _print_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def _resolve_level_ids(
    catalog: LevelCatalog, args: argparse.Namespace
) -> tuple[str, ...]:
    if getattr(args, "level", None):
        return tuple(args.level)
    if getattr(args, "held_out", False):
        return tuple(level.id for level in catalog.held_out_levels)
    if getattr(args, "all_levels", False):
        return tuple(level.id for level in catalog.train_levels)
    return (catalog[0].id,)


def _auto_workers(requested: str | int | None, num_envs: int) -> int:
    if requested in (None, "auto"):
        cpus = os.cpu_count() or 1
        # Small machines measured faster with the in-process adapter; larger machines
        # benefit from one worker per environment.
        return 1 if cpus <= 4 else min(num_envs, max(2, cpus // 2))
    value = int(requested)
    if value < 1:
        raise ValueError("workers must be >= 1")
    return value


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def command_doctor(args: argparse.Namespace) -> int:
    import platform

    import mujoco
    import numpy

    print(f"AI Parkour Lab {__version__}")
    print(f"python      {platform.python_version()} ({platform.platform()})")
    print(f"numpy       {numpy.__version__}")
    print(f"mujoco      {mujoco.__version__}")
    print(f"cpu count   {os.cpu_count()}")
    try:
        import torch  # type: ignore

        print(f"torch       {torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            print(f"gpu         {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch       not installed (optional; not required for the CPU/NumPy stack)")
    print(f"MUJOCO_GL   {os.environ.get('MUJOCO_GL', 'unset')}")
    catalog = _catalog(args)
    print(f"levels      {len(catalog)} ({len(catalog.train_levels)} train, "
          f"{len(catalog.held_out_levels)} held-out) in {DEFAULT_LEVELS_DIR}")
    try:
        level = catalog[0]
        env = ParkourEnv(level, EnvConfig(level_id=level.id))
        print(
            f"model       {env.model.nq} qpos, {env.model.nv} dof, {env.model.nu} actuators, "
            f"{env.model.ngeom} geoms, mass {sum(env.model.body_mass):.1f} kg"
        )
        print(f"observation {env.observation_dim} values, action {env.action_dim} values")
        env.close()
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"model       FAILED: {type(exc).__name__}: {exc}")
        return 1
    import subprocess

    for backend in ("egl", "osmesa"):
        probe = (
            "import mujoco; m=mujoco.MjModel.from_xml_string("
            "'<mujoco><worldbody><geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>');"
            "r=mujoco.Renderer(m,32,32); r.update_scene(mujoco.MjData(m)); r.render()"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**os.environ, "MUJOCO_GL": backend},
            timeout=60,
        )
        if result.returncode == 0:
            print(f"offscreen   {backend}: available")
            break
    else:
        print(
            "offscreen   no EGL/OSMesa software renderer; the native viewer needs a display "
            "and video recording is unavailable. Headless training is unaffected."
        )
    return 0


def command_levels(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    if args.json:
        print(json.dumps([level.to_dict() for level in catalog], indent=2))
        return 0
    rows = []
    for level in catalog:
        rows.append(
            [
                level.order,
                level.id,
                level.difficulty,
                level.split,
                f"{level.finish_position.x - level.spawn_position.x:.1f}",
                level.time_limit_seconds,
                len(level.obstacles),
                level.title,
            ]
        )
    _print_table(
        rows,
        ["#", "id", "diff", "split", "length", "limit(s)", "obstacles", "title"],
    )
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    level = catalog[args.level]
    spec = default_humanoid()
    print(f"level {level.id}: {level.title} (difficulty {level.difficulty})")
    print(f"  spawn   {level.spawn_position.as_tuple()} -> finish {level.finish_position.as_tuple()}")
    print(f"  time limit {level.time_limit_seconds}s, par {level.par_time_seconds}s")
    print(f"  obstacles ({len(level.obstacles)}):")
    for obstacle in level.obstacles:
        extra = ""
        if obstacle.hazard:
            extra += " hazard"
        if obstacle.motion:
            extra += (
                f" motion axis={obstacle.motion.axis.as_tuple()} amp={obstacle.motion.amplitude}"
                f" period={obstacle.motion.period_seconds}s"
            )
        print(
            f"    {obstacle.id:<20} {obstacle.kind:<16} pos={obstacle.position.as_tuple()} "
            f"size={obstacle.size.as_tuple()} top={obstacle.top_z:.2f}{extra}"
        )
    env = ParkourEnv(level, EnvConfig(level_id=level.id, action_scale=args.action_scale))
    try:
        print("\nhumanoid inventory")
        print(f"  total mass {spec.total_mass:.2f} kg over {len(spec.bodies)} bodies")
        bounds = np.degrees(env.action_bounds())
        for group, joints in spec.by_group().items():
            print(f"  {group}: {len(joints)} joints")
            for index, joint in enumerate(joints):
                slot = env.humanoid.joint_names.index(joint.name)
                reachable = bounds[slot]
                print(
                    f"    {joint.name:<20} range=[{joint.lower:7.1f}, {joint.upper:6.1f}] deg "
                    f"reach=[{reachable[0]:6.1f}, {reachable[1]:6.1f}] travel={reachable[1] - reachable[0]:5.1f} deg "
                    f"kp={joint.kp:5.0f} force={joint.force_limit:5.0f} default={joint.default:5.1f}"
                )
        print(f"\nobservation: {len(OBSERVATION_NAMES)} task values + 3x{ACTION_DIM} proprio")
        print("  " + ", ".join(OBSERVATION_NAMES[:12]) + ", ...")
        print(
            f"action: {env.action_dim} clamped joint targets in [-1, 1] around the default pose, "
            f"action_scale={args.action_scale:.2f}"
        )
        travel = np.diff(bounds, axis=1).ravel()
        knee = env.humanoid.joint_names.index("knee_l")
        print(
            f"  reachable travel per joint: {travel.min():.1f}-{travel.max():.1f} deg "
            f"(knee_l {travel[knee]:.1f} deg of its "
            f"{spec.joints[knee].upper - spec.joints[knee].lower:.0f} deg range). "
            "The box is default +/- scale * half_range; raise --action-scale (or use "
            "action_mode='range') for more authority."
        )
        print(f"model: nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
              f"timestep={env.model.opt.timestep}")
    finally:
        env.close()
    return 0


def command_export_mjcf(args: argparse.Namespace) -> int:
    from .course import build_course

    catalog = _catalog(args)
    level = catalog[args.level]
    scene = build_course(level) if args.course else None
    text = scene.xml if scene is not None else humanoid_xml(default_humanoid())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} bytes)")
    return 0


def command_random(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    level = catalog[args.level]
    env = ParkourEnv(level, EnvConfig(level_id=level.id, seed=args.seed))
    rng = np.random.default_rng(args.seed)
    observation, info = env.reset(seed=args.seed)
    total_reward = 0.0
    steps = 0
    while steps < args.steps:
        action = np.zeros(env.action_dim) if args.policy == "stand" else rng.uniform(
            -1.0, 1.0, env.action_dim
        )
        result = env.step(action)
        total_reward += result.reward
        steps += 1
        if result.terminated or result.truncated:
            break
    summary = env.episode_summary()
    print(json.dumps(summary, indent=2))
    print(f"steps={steps} reward={total_reward:.3f} terminated={env.termination_reason}")
    env.close()
    return 0


def _parse_reward_overrides(specs: Sequence[str] | None, *, flag: str) -> dict[str, float]:
    from .env import RewardConfig

    overrides: dict[str, float] = {}
    for spec in specs or []:
        key, separator, raw = spec.partition("=")
        key = key.strip()
        if not separator or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {spec!r}")
        if key not in RewardConfig.__dataclass_fields__:
            raise SystemExit(f"unknown reward weight {key!r}")
        try:
            overrides[key] = float(raw)
        except ValueError as error:
            raise SystemExit(f"{flag} {key}: {error}") from error
    return overrides


def _build_train_config(args: argparse.Namespace, catalog: LevelCatalog) -> TrainConfig:
    level_ids = _resolve_level_ids(catalog, args)
    curriculum = args.curriculum
    if curriculum == "auto":
        curriculum = "single" if len(level_ids) == 1 else "adaptive"
    if curriculum == "single" and len(level_ids) != 1:
        raise SystemExit("--curriculum single needs exactly one --level")
    if args.eval_levels:
        eval_levels = tuple(args.eval_levels)
    elif args.all_levels:
        eval_levels = tuple(
            sorted(
                {
                    level_ids[int(round(index))]
                    for index in np.linspace(0, len(level_ids) - 1, min(4, len(level_ids)))
                }
            )
        )
    else:
        eval_levels = level_ids
    num_envs = max(1, args.envs)
    workers = _auto_workers(args.workers, num_envs)
    ppo_overrides: dict[str, Any] = {}
    if args.lr is not None:
        ppo_overrides["learning_rate"] = args.lr
    if args.epochs is not None:
        ppo_overrides["epochs"] = args.epochs
    if args.entropy is not None:
        ppo_overrides["entropy_coefficient"] = args.entropy
    if args.gamma is not None:
        ppo_overrides["gamma"] = args.gamma
    from .ppo import PPOConfig

    ppo = PPOConfig(**ppo_overrides) if ppo_overrides else PPOConfig()
    curriculum_config = CurriculumConfig()
    from .env import RewardConfig

    reward = RewardConfig(**_parse_reward_overrides(args.reward_set, flag="--reward-set"))
    reward_start: RewardConfig | None = None
    ramp = float(getattr(args, "reward_ramp_fraction", 0.0) or 0.0)
    if ramp > 0.0:
        start_overrides = _parse_reward_overrides(args.reward_start, flag="--reward-start")
        reward_start = replace(RewardConfig.posture_stage(), **start_overrides)
    return TrainConfig(
        level_ids=level_ids,
        curriculum=curriculum,
        seed=args.seed,
        total_steps=args.steps,
        rollout_steps=args.rollout_steps,
        physics_timestep=args.physics_timestep,
        control_decimation=args.decimation,
        num_envs=num_envs,
        workers=workers,
        ppo=ppo,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        initial_log_std=args.init_log_std,
        actuation=args.actuation,
        eval_interval=args.eval_interval,
        eval_levels=eval_levels,
        eval_seeds=tuple(args.eval_seeds),
        eval_episodes_per_level=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        keep_checkpoints=args.keep_checkpoints,
        max_minutes=args.max_minutes,
        target_success=args.target_success,
        reward_scale=args.reward_scale,
        reward=reward,
        reward_start=reward_start,
        reward_ramp_fraction=ramp,
        max_episode_seconds=args.max_episode_seconds,
        stagnation_seconds=args.stagnation_seconds,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr if args.lr is not None else 3e-4,
        name=args.name or "",
        notes=args.notes or "",
        curriculum_config=curriculum_config,
        max_replays=args.max_replays,
        record_replay_every_eval=args.replay_every,
    )


def command_train(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    config = _build_train_config(args, catalog)
    print(f"training {len(config.level_ids)} level(s), curriculum={config.curriculum}, "
          f"envs={config.num_envs}, workers={config.workers}, steps={config.total_steps:,}")
    if config.actuation == "muscle":
        print("note: muscle actuation uses activation-filtered torque control (experimental)")
    summary = train(
        config,
        catalog,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        resume_from=args.resume,
    )
    print(json.dumps({key: summary[key] for key in (
        "run_id", "status", "total_steps", "iterations", "episodes",
        "best_eval_success", "best_eval_completion", "wall_seconds",
    ) if key in summary}, indent=2))
    return 0


def _env_config_from_checkpoint(extra: dict[str, Any], level_id: str) -> EnvConfig:
    """Rebuild the exact training dynamics recorded with a checkpoint."""

    return env_config_from_checkpoint(extra, level_id)


def _load_policy(args: argparse.Namespace) -> ActorCritic:
    model, extra = ActorCritic.load(args.checkpoint, seed=getattr(args, "seed", 0))
    if not hasattr(args, "_checkpoint_extra"):
        args._checkpoint_extra = extra  # type: ignore[attr-defined]
    return model


def _checkpoint_env_config(args: argparse.Namespace, level_id: str) -> EnvConfig:
    extra = getattr(args, "_checkpoint_extra", None)
    if extra and extra.get("config"):
        return _env_config_from_checkpoint(extra, level_id)
    return EnvConfig(level_id=level_id)


def command_diagnose(args: argparse.Namespace) -> int:
    """Explain what a checkpoint's policy actually does, one traced episode per seed."""

    catalog = _catalog(args)
    model = _load_policy(args)
    extra = getattr(args, "_checkpoint_extra", None) or {}
    recorded = [str(level_id) for level_id in extra.get("level_ids") or []]
    if args.level:
        level_id = args.level
    elif recorded:
        level_id = recorded[0]
    else:
        level_id = catalog.train_levels[0].id
    level = catalog[level_id]
    env_config = _checkpoint_env_config(args, level.id)
    seeds = tuple(args.eval_seeds) if args.eval_seeds else EvalConfig().seeds
    diagnosis = diagnose_policy(
        model,
        level,
        seeds=seeds,
        env_config=env_config,
        deterministic=not args.stochastic,
    )
    if args.json:
        print(json.dumps(_jsonable(diagnosis), indent=2))
    else:
        print(f"checkpoint: {args.checkpoint}")
        print(format_diagnosis(diagnosis))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    model = _load_policy(args)
    level_ids = _resolve_level_ids(catalog, args)
    result = evaluate_policy(
        model,
        catalog,
        level_ids,
        config=EvalConfig(
            seeds=tuple(args.eval_seeds),
            deterministic=not args.stochastic,
            record_frames=args.record_replay,
        ),
        env_config=_checkpoint_env_config(args, level_ids[0]),
    )
    print(format_evaluation_table(result))
    print("\ntermination reasons:", json.dumps(result["aggregate"]["terminations"]))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        if args.record_replay:
            for record in payload["records"]:
                record.pop("frames", None)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    if args.record_replay and args.replay_out:
        directory = Path(args.replay_out)
        directory.mkdir(parents=True, exist_ok=True)
        import gzip

        for record in result["records"]:
            frames = record.pop("frames", [])
            if not frames:
                continue
            payload = {
                "level_id": record["level_id"],
                "level_title": record.get("level_title"),
                "seed": record.get("seed"),
                "success": record.get("success"),
                "termination_reason": record.get("termination_reason"),
                "completion_fraction": record.get("completion_fraction"),
                "frames": frames,
                "level": catalog[record["level_id"]].to_dict(),
            }
            path = directory / f"{record['level_id']}-seed{record.get('seed')}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            print(f"wrote replay {path}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    store_a = RunStore.load(Path(args.run_a) if Path(args.run_a).is_dir() else Path(args.runs_dir) / args.run_a)
    store_b = RunStore.load(Path(args.run_b) if Path(args.run_b).is_dir() else Path(args.runs_dir) / args.run_b)
    comparison = compare_runs(store_a, store_b)
    if "error" in comparison:
        print(comparison["error"], file=sys.stderr)
        return 1
    rows = [
        [
            row["level_id"],
            f"{row['baseline_success'] * 100:5.1f}%",
            f"{row['candidate_success'] * 100:5.1f}%",
            f"{row['delta_success'] * 100:+5.1f}%",
            f"{row['baseline_completion'] * 100:5.1f}%",
            f"{row['candidate_completion'] * 100:5.1f}%",
        ]
        for row in comparison["rows"]
    ]
    _print_table(
        rows,
        ["level", "A success", "B success", "delta", "A completion", "B completion"],
    )
    overall = comparison["overall"]
    print(
        f"\nA={comparison['run_a']['run_id']} B={comparison['run_b']['run_id']}: "
        f"success {overall['delta_success_rate'] * 100:+.1f} pp, "
        f"completion {overall['delta_mean_completion'] * 100:+.1f} pp"
    )
    return 0


def command_runs(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_dir)
    if getattr(args, "json", False):
        print(json.dumps({"runs": _jsonable(runs)}, indent=2))
        return 0
    if not runs:
        print(f"no runs in {args.runs_dir}")
        return 0
    rows = []
    for run in runs:
        summary = run["summary"]
        progress = run.get("progress_steps") or 0
        target = run.get("target_steps") or run.get("total_steps") or 0
        best = run.get("best_eval_success")
        if best is None:
            best = summary.get("best_eval_success")
        rows.append(
            [
                run["run_id"],
                run["status"],
                f"{progress:,} / {target:,}" if target else f"{progress:,}",
                f"{float(best) * 100:.1f}%" if best is not None else "",
                len(run["checkpoints"]),
                run["replay_count"],
                (summary.get("updated_utc") or "")[:19],
            ]
        )
    _print_table(rows, ["run", "status", "steps", "best eval", "ckpts", "replays", "updated"])
    return 0


def _jsonable(value: Any) -> Any:
    """Convert Path/NumPy values produced by storage into plain JSON types."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def command_benchmark(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    levels = _resolve_level_ids(catalog, args)
    env_counts = [int(value) for value in args.envs.split(",") if value.strip()]
    worker_counts = (
        [int(value) for value in args.worker_list.split(",") if value.strip()]
        if args.worker_list
        else [1, 0]  # 0 means "one worker per environment" (the subprocess layout)
    )
    results = []
    print(f"benchmarking {args.steps} control steps per configuration...")
    for requested in worker_counts:
        for num_envs in env_counts:
            if num_envs > len(levels):
                level_ids = [levels[index % len(levels)] for index in range(num_envs)]
            else:
                level_ids = list(levels[:num_envs])
            workers = num_envs if requested <= 0 else requested
            if workers > num_envs:
                continue
            result = benchmark_simulation(
                catalog,
                levels=level_ids,
                num_envs=num_envs,
                steps=args.steps,
                workers=workers,
                env_config=EnvConfig(level_id=level_ids[0]),
            )
            results.append(result)
            print(
                f"  envs={num_envs} workers={workers}: "
                f"{result['steps_per_second']:.0f} transitions/s "
                f"({result['elapsed_seconds']:.2f}s)"
            )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    best = max(results, key=lambda result: result["steps_per_second"])
    print(
        f"best measured configuration: envs={best['num_envs']} workers={best['workers']} "
        f"-> {best['steps_per_second']:.0f} transitions/s"
    )
    return 0


def command_watch(args: argparse.Namespace) -> int:
    """Interactive MuJoCo viewer playback of a checkpoint (needs a display/GPU)."""

    catalog = _catalog(args)
    level = catalog[args.level]
    try:
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover - depends on the platform
        print(f"MuJoCo viewer unavailable: {exc}", file=sys.stderr)
        return 1
    model = None
    if args.checkpoint:
        model = _load_policy(args)
    env = ParkourEnv(level, EnvConfig(level_id=level.id, seed=args.seed))
    observation, _ = env.reset(seed=args.seed, options={"deterministic": True})
    rng = np.random.default_rng(args.seed)
    control_dt = env.config.control_dt
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 4.5
            viewer.cam.elevation = -12
            viewer.cam.azimuth = 130
            while viewer.is_running():
                start = time.perf_counter()
                if model is None:
                    action = np.zeros(env.action_dim) if args.policy == "stand" else rng.uniform(
                        -1, 1, env.action_dim
                    )
                else:
                    action = model.act(observation, deterministic=not args.stochastic)
                result = env.step(action)
                observation = result.observation
                if result.terminated or result.truncated:
                    print(
                        f"episode ended: {result.info['termination_reason']} "
                        f"progress={result.info['progress_fraction'] * 100:.1f}% "
                        f"return={env.episode_return:.2f}",
                        flush=True,
                    )
                    if args.once:
                        break
                    observation, _ = env.reset(options={"deterministic": True})
                if not args.fast:
                    elapsed = time.perf_counter() - start
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
            viewer.close()
    except Exception as exc:  # pragma: no cover - headless machines
        print(
            f"could not open the MuJoCo viewer ({type(exc).__name__}: {exc}).\n"
            "The viewer needs a display and an OpenGL context. Use "
            "`parkour dashboard` and the browser 3D replay instead on headless machines.",
            file=sys.stderr,
        )
        env.close()
        return 1
    env.close()
    return 0


def command_dashboard(args: argparse.Namespace) -> int:
    from .dashboard.server import run_dashboard

    return run_dashboard(
        runs_dir=Path(args.runs_dir),
        host=args.host,
        port=args.port,
        levels_dir=Path(args.levels_dir) if args.levels_dir else None,
        open_browser=args.open,
    )


def command_replay(args: argparse.Namespace) -> int:
    """Re-simulate and export one episode from a checkpoint."""

    catalog = _catalog(args)
    level_id = args.level or catalog[0].id
    level = catalog[level_id]
    model = _load_policy(args)
    record = rollout_episode(
        model,
        level,
        seed=args.seed,
        env_config=_checkpoint_env_config(args, level_id),
        deterministic=not args.stochastic,
        record_frames=True,
        frame_stride=args.frame_stride,
    )
    frames = record.pop("frames", [])
    print(json.dumps({key: value for key, value in record.items() if key != "initial_info"}, indent=2))
    payload = {
        "run_id": Path(args.checkpoint).parent.parent.name,
        "level_id": level_id,
        "level_title": level.title,
        "seed": args.seed,
        "success": record["success"],
        "termination_reason": record["termination_reason"],
        "completion_fraction": record["completion_fraction"],
        "frames": frames,
        "level": level.to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gz":
        import gzip

        with gzip.open(out, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    else:
        out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote replay with {len(frames)} frames to {out}")
    return 0


def command_evolve(args: argparse.Namespace) -> int:
    from .training.evolution import EvolutionConfig, evolve_reward_weights

    catalog = _catalog(args)
    level_ids = _resolve_level_ids(catalog, args)
    config = EvolutionConfig(
        generations=args.generations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        train_steps=args.train_steps,
        num_envs=args.envs,
        workers=_auto_workers(args.workers, args.envs),
        seed=args.seed,
        eval_seeds=tuple(args.eval_seeds),
    )
    result = evolve_reward_weights(
        catalog,
        list(level_ids),
        config=config,
        runs_dir=args.runs_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"artifacts: {result['run_dir']}")
    return 0


def command_harness(args: argparse.Namespace) -> int:
    from .harness2 import AssistantConfig, ResearchAssistant

    catalog = _catalog(args)
    if args.action == "status":
        print(json.dumps(_harness_status(), indent=2))
        return 0
    assistant = ResearchAssistant(
        catalog,
        AssistantConfig(
            min_improvement=args.min_improvement,
            max_proposals=args.max_proposals,
            allow_network=args.allow_network,
        ),
    )
    if args.action == "plan":
        proposals = assistant.plan(
            run_dir=Path(args.run) if args.run else None,
            target_levels=args.level or None,
        )
        print(json.dumps([proposal.to_dict() for proposal in proposals], indent=2))
        return 0
    if args.action == "probe":
        if not args.run:
            print("--run is required for probe", file=sys.stderr)
            return 1
        report = assistant.probe(Path(args.run), runs_dir=args.runs_dir)
        print(json.dumps(report.to_dict(), indent=2))
        print(f"report: {report.report_path}", file=sys.stderr)
        return 0
    raise SystemExit(f"unknown harness action {args.action!r}")


def _harness_status() -> dict[str, Any]:
    return {
        "name": "Harness²-style restricted research assistant",
        "role": "experiment planning, bounded config proposals, evaluation and comparison",
        "cannot": [
            "simulate physics",
            "control the humanoid",
            "edit or train policy weights",
            "promote a checkpoint",
            "change level geometry",
        ],
        "edit_surface": [
            "reward weights (bounded ranges)",
            "PPO learning rate, entropy coefficient, gamma (bounded ranges)",
            "action scale and training step budget (bounded ranges)",
        ],
        "network": "disabled unless --allow-network is passed explicitly",
    }


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parkour",
        description="AI Parkour Lab: MuJoCo humanoid parkour training and research tools.",
    )
    parser.add_argument("--version", action="version", version=f"ai-parkour-lab {__version__}")
    parser.add_argument(
        "--runs-dir", default=str(DEFAULT_RUNS_DIR), help="directory holding run records"
    )
    parser.add_argument(
        "--levels-dir", default=str(DEFAULT_LEVELS_DIR), help="directory holding level manifests"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check the local environment")
    doctor.set_defaults(func=command_doctor)

    levels = subparsers.add_parser("list-levels", help="list fixed parkour levels")
    levels.add_argument("--json", action="store_true")
    levels.set_defaults(func=command_levels)

    inspect = subparsers.add_parser("inspect", help="inspect a level and the humanoid")
    inspect.add_argument("--level", default=None, help="level id (default: first level)")
    inspect.add_argument(
        "--action-scale",
        dest="action_scale",
        type=float,
        default=0.25,
        help="action box scale used for the reachable-range column (default 0.25)",
    )
    inspect.set_defaults(func=command_inspect)

    export = subparsers.add_parser("export-mjcf", help="export the MJCF scene for a level")
    export.add_argument("--level", default=None)
    export.add_argument("--out", required=True)
    export.add_argument("--course", action="store_true", default=True)
    export.set_defaults(func=command_export_mjcf)

    random_cmd = subparsers.add_parser("random", help="run a random or standing policy")
    random_cmd.add_argument("--level", default=None)
    random_cmd.add_argument("--seed", type=int, default=0)
    random_cmd.add_argument("--steps", type=int, default=500)
    random_cmd.add_argument("--policy", choices=["stand", "random"], default="random")
    random_cmd.set_defaults(func=command_random)

    train_cmd = subparsers.add_parser("train", help="train a PPO policy")
    train_cmd.add_argument("--level", action="append", help="level id (repeatable)")
    train_cmd.add_argument("--all", dest="all_levels", action="store_true",
                           help="train on every training level")
    train_cmd.add_argument("--held-out", action="store_true",
                           help="train on held-out levels (normally evaluation-only)")
    train_cmd.add_argument("--curriculum", choices=["auto", "adaptive", "uniform", "single"],
                           default="auto")
    train_cmd.add_argument("--steps", type=int, default=1_000_000)
    train_cmd.add_argument("--rollout-steps", type=int, default=256)
    train_cmd.add_argument(
        "--physics-timestep", type=float, default=0.002, help="MuJoCo integration step (s)"
    )
    train_cmd.add_argument(
        "--decimation", type=int, default=5, help="physics steps per control step"
    )
    train_cmd.add_argument("--envs", type=int, default=4)
    train_cmd.add_argument("--workers", default="auto")
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--lr", type=float, default=None)
    train_cmd.add_argument("--epochs", type=int, default=None)
    train_cmd.add_argument("--entropy", type=float, default=None)
    train_cmd.add_argument("--gamma", type=float, default=None)
    train_cmd.add_argument("--hidden-dim", type=int, default=128)
    train_cmd.add_argument("--action-scale", type=float, default=0.25)
    train_cmd.add_argument(
        "--init-log-std", type=float, default=-1.0, help="initial Gaussian action noise (log std)"
    )
    train_cmd.add_argument(
        "--action-smoothing",
        type=float,
        default=0.6,
        help="first-order filter on actuator targets (0 = raw, higher = smoother)",
    )
    train_cmd.add_argument("--actuation", choices=["position", "muscle"], default="position")
    train_cmd.add_argument("--reward-scale", type=float, default=0.25)
    train_cmd.add_argument(
        "--reward-set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override one reward weight, repeatable (e.g. --reward-set height_per_second=2.0)",
    )
    train_cmd.add_argument(
        "--reward-start",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "override a weight of the posture-only starting reward used with "
            "--reward-ramp-fraction"
        ),
    )
    train_cmd.add_argument(
        "--reward-ramp-fraction",
        type=float,
        default=0.0,
        help=(
            "anneal the reward from a posture-only stage to the final weights over this "
            "fraction of the run (0 disables)"
        ),
    )
    train_cmd.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help="cap episode length below the level time limit (seconds)",
    )
    train_cmd.add_argument(
        "--stagnation-seconds",
        type=float,
        default=0.0,
        help="end an episode after this long without approaching the finish (0 disables)",
    )
    train_cmd.add_argument("--eval-interval", type=int, default=20)
    train_cmd.add_argument("--eval-levels", action="append", default=None)
    train_cmd.add_argument("--eval-seeds", type=int, nargs="+", default=[11, 23])
    train_cmd.add_argument("--eval-episodes", type=int, default=1)
    train_cmd.add_argument("--checkpoint-interval", type=int, default=20)
    train_cmd.add_argument("--keep-checkpoints", type=int, default=6)
    train_cmd.add_argument("--max-replays", type=int, default=12)
    train_cmd.add_argument("--replay-every", type=int, default=2)
    train_cmd.add_argument("--max-minutes", type=float, default=None)
    train_cmd.add_argument("--target-success", type=float, default=None)
    train_cmd.add_argument("--run-id", default=None)
    train_cmd.add_argument("--name", default=None)
    train_cmd.add_argument("--notes", default=None)
    train_cmd.add_argument("--resume", default=None, help="checkpoint path to resume from")
    train_cmd.set_defaults(func=command_train)

    diagnose = subparsers.add_parser(
        "diagnose", help="trace episodes and explain why a checkpoint fails"
    )
    diagnose.add_argument("--checkpoint", required=True)
    diagnose.add_argument("--level", default=None)
    diagnose.add_argument("--eval-seeds", nargs="+", type=int, default=None)
    diagnose.add_argument("--stochastic", action="store_true")
    diagnose.add_argument("--seed", type=int, default=0)
    diagnose.add_argument("--json", action="store_true")
    diagnose.set_defaults(func=command_diagnose)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint deterministically")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--level", action="append")
    evaluate.add_argument("--all", dest="all_levels", action="store_true")
    evaluate.add_argument("--held-out", action="store_true")
    evaluate.add_argument("--eval-seeds", type=int, nargs="+", default=[11, 23, 37])
    evaluate.add_argument("--stochastic", action="store_true")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--out", default=None)
    evaluate.add_argument("--record-replay", action="store_true")
    evaluate.add_argument("--replay-out", default=None)
    evaluate.set_defaults(func=command_evaluate)

    replay = subparsers.add_parser("replay", help="record an episode as a 3D replay file")
    replay.add_argument("--checkpoint", required=True)
    replay.add_argument("--level", default=None)
    replay.add_argument("--seed", type=int, default=11)
    replay.add_argument("--stochastic", action="store_true")
    replay.add_argument("--frame-stride", type=int, default=2)
    replay.add_argument("--out", required=True)
    replay.set_defaults(func=command_replay)

    watch = subparsers.add_parser("watch", help="open the native MuJoCo viewer")
    watch.add_argument("--checkpoint", default=None)
    watch.add_argument("--level", default=None)
    watch.add_argument("--seed", type=int, default=7)
    watch.add_argument("--policy", choices=["stand", "random"], default="stand")
    watch.add_argument("--stochastic", action="store_true")
    watch.add_argument("--once", action="store_true", help="stop after the first episode")
    watch.add_argument("--fast", action="store_true", help="do not throttle to real time")
    watch.set_defaults(func=command_watch)

    dashboard = subparsers.add_parser("dashboard", help="serve the research dashboard")
    dashboard.add_argument("--host", default="0.0.0.0")
    dashboard.add_argument("--port", type=int, default=8000)
    dashboard.add_argument("--open", action="store_true")
    dashboard.set_defaults(func=command_dashboard)

    benchmark = subparsers.add_parser("benchmark", help="benchmark parallel environments")
    benchmark.add_argument("--level", action="append")
    benchmark.add_argument("--all", dest="all_levels", action="store_true")
    benchmark.add_argument("--envs", default="1,2,4")
    benchmark.add_argument(
        "--workers",
        dest="worker_list",
        default=None,
        help="comma-separated worker counts; default measures 1 and one-per-env",
    )
    benchmark.add_argument("--steps", type=int, default=200)
    benchmark.add_argument("--out", default=None)
    benchmark.set_defaults(func=command_benchmark)

    runs = subparsers.add_parser("runs", help="list local runs")
    runs.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    runs.set_defaults(func=command_runs)

    compare = subparsers.add_parser("compare", help="compare two runs' best evaluations")
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)
    compare.set_defaults(func=command_compare)

    evolve = subparsers.add_parser("evolve", help="bounded CEM over reward/PPO parameters")
    evolve.add_argument("--level", action="append")
    evolve.add_argument("--all", dest="all_levels", action="store_true")
    evolve.add_argument("--generations", type=int, default=3)
    evolve.add_argument("--population", type=int, default=4)
    evolve.add_argument("--elite-fraction", type=float, default=0.5)
    evolve.add_argument("--train-steps", type=int, default=60_000)
    evolve.add_argument("--envs", type=int, default=2)
    evolve.add_argument("--workers", default="auto")
    evolve.add_argument("--seed", type=int, default=0)
    evolve.add_argument("--eval-seeds", type=int, nargs="+", default=[11])
    evolve.set_defaults(func=command_evolve)

    harness = subparsers.add_parser(
        "harness", help="restricted Harness²-style research assistant (planning only)"
    )
    harness.add_argument("action", choices=["status", "plan", "probe"])
    harness.add_argument("--run", default=None)
    harness.add_argument("--level", action="append")
    harness.add_argument("--max-proposals", type=int, default=3)
    harness.add_argument("--min-improvement", type=float, default=0.02)
    harness.add_argument("--allow-network", action="store_true")
    harness.set_defaults(func=command_harness)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LevelError as exc:
        print(f"level error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
