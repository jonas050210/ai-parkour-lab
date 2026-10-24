# Architecture

This document describes the system as implemented, not as planned. It separates
**simulation**, **control/learning**, **evaluation**, **persistence** and **presentation**, and
names the module that owns each part.

## Component map

```text
configs/levels/*.json ──> levels.LevelCatalog ──> level.Level (validated manifest)
                                    │
humanoid.HumanoidSpec (25 DoF) ─────┤
                                    v
                         course.build_course(level) ──> one MJCF scene string
                                    │                        (course + humanoid + finish + hazards)
                                    v
                    env.ParkourEnv  (Gymnasium-shaped: reset / step / episode_summary)
                       │      │             │
        observations   │      │ reward      │ frames (JSON, no rendering)
                       v      v             v
       training.vector SerialVecEnv / SubprocVecEnv  ──> training.runner.TrainingRunner
                       │                                        │
                       │                                        ├─> storage.RunStore  (runs/<id>/…)
                       │                                        └─> evaluation.evaluate_policy
                       v
             policies.ActorCritic + ppo.ppo_update (NumPy)
                       │
             ┌─────────┴──────────┐
             v                    v
  training.evolution      harness2.ResearchAssistant
  (CEM over 8 bounded      (11 bounded knobs; proposals only)
   knobs, real PPO runs)
                                    │
                                    v
                    dashboard.server (FastAPI, read-only + explicit
                    local train start/stop) ──> static/app.js (Three.js viewer)
```

## Modules and responsibilities

| Module | Owns | Must not own |
|---|---|---|
| `levels.py` | Manifest loading, schema validation, `LevelCatalog`, train/held-out split | Physics, geometry generation |
| `humanoid.py` | The articulated humanoid XML: bodies, 19 actuated joints, geometry, position/muscle actuators | Task logic, rewards |
| `course.py` | Compiling a level + humanoid into one MJCF document, moving platforms, finish marker, hazards | Policy, evaluation |
| `env.py` | Reset/step contract, observation and action mapping, reward components, termination, contact diagnostics, replay frame capture | Optimiser, storage, HTTP |
| `policies.py` | MLP actor/critic, observation normalisation, action distribution, checkpoint save/load | Environment dynamics, evaluation protocol |
| `ppo.py` | GAE, clipped surrogate update, schedule | Data collection |
| `training/vector.py` | Serial and subprocess vectorised envs, level switching, `benchmark_simulation` | Learning |
| `training/curriculum.py` | Level sampler (`single`, `uniform`, `adaptive`) and per-level performance summaries | Reward weights |
| `training/runner.py` | Training loop, checkpointing, periodic evaluation, replays, CSV/JSONL metrics | Level authoring, UI |
| `evaluation.py` | Deterministic/stochastic rollouts, aggregation, comparison, checkpoint-dynamics recovery | Training updates |
| `diagnostics.py` | Traced rollouts (speed, pelvis height, per-foot contact) and named findings for a checkpoint | Training updates |
| `storage.py` | Run directory layout, atomic JSON writes, run listing, provenance, deletion | Simulation |
| `dashboard/server.py` | Read-only API over run directories, validated local train start/stop, static files | Physics simulation, model edits |
| `training/evolution.py` | Bounded CEM over 8 declared parameters; every candidate is a real PPO run | Weight search, physics changes |
| `harness2.py` | Rule-based proposals over 11 bounded knobs, probe reports | Applying changes, promoting checkpoints |

`contracts.py` keeps the dependency-free `Environment`/`Policy` protocols and the frozen
`EpisodeSummary` used by the foundation tests; the concrete implementation lives in the modules
above and is free to depend on NumPy and MuJoCo.

## Data flow

1. `parkour train` (or the dashboard) builds a `TrainConfig`, which resolves levels through
   `LevelCatalog` and produces an `EnvConfig` per environment.
2. `make_vec_env` creates either serial or subprocess environments. Each environment compiles its
   own MJCF scene from the level manifest; levels are cached per slot and swapped between episodes.
3. Each control step: observation (90 values) → policy → 19 normalised actions → centred mapping onto
   radian joint targets → `control_decimation` MuJoCo steps → reward components, termination check,
   optional replay frame.
4. `TrainingRunner` collects `rollout_steps × num_envs` transitions, computes GAE, updates the
   policy, appends a CSV row, records completed episodes as JSONL, and periodically evaluates,
   checkpoints and records a replay.
5. `RunStore` writes everything under `runs/<run-id>/` with atomic JSON writes; a `summary.json`
   carries status and best-evaluation bookkeeping.
6. The dashboard reads those directories (with an mtime-keyed cache), serves JSON, and can spawn one
   local `parkour train` process at a time, streaming its log.
7. The browser reconstructs bodies, courses and replays from JSON; it never steps physics.

## Run directory layout (implemented)

```text
runs/<run-id>/
  config.json              exact TrainConfig, including reward start weights and ramp fraction
  provenance.json          git revision/dirty flag, package versions, CPU/GPU, MUJOCO_GL
  summary.json             status, steps, iterations, best evaluation, level ids
  metrics/training.csv     one row per iteration (loss, entropy, KL, throughput, eval columns)
  metrics/episodes.jsonl   one record per finished episode (level, seed, reason, completion, …)
  metrics/evaluation.csv   one row per periodic evaluation
  evaluation/iter_XXXXX.json  full evaluation payload (records + aggregate)
  checkpoints/latest.npz   policy + normaliser + optimiser-free extras; `best.npz` and `iter_*.npz`
  replays/iterXXXXX-level-seed.json.gz  bounded replay frames for the dashboard
  harness2/probe-*.json    research-assistant probe reports
  logs/train.log           stdout of a dashboard-launched training job
```

## Key boundaries

- **Physics vs learning.** `env.py` is the only module that calls MuJoCo. The learner sees arrays.
- **Training vs evaluation.** Evaluation loads a checkpoint read-only, rebuilds the recorded
  dynamics (`evaluation.env_config_from_checkpoint`) and never updates the policy.
- **Simulation vs presentation.** The dashboard never simulates; replays are JSON frames captured
  during a real rollout, so a headless-trained agent is replayable in the browser on a machine with
  no GPU or display.
- **Learner vs assistants.** `training/evolution.py` and `harness2.py` can only propose bounded
  numbers. They cannot edit physics, observation, level geometry, or apply changes automatically;
  `tests/test_research_tools.py` enforces this.
- **Single source of truth for a run.** Everything a result depends on is recorded in
  `config.json` + `provenance.json` + the checkpoint's `extra` block, so a run can be rebuilt
  without this document.
