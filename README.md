# AI Parkour Lab

A local-first platform for training a physically simulated humanoid to traverse fixed parkour
courses. Everything runs on a CPU-only machine: MuJoCo for physics, NumPy PPO for learning,
a CLI for training/evaluation/replay, and a local browser dashboard for inspecting runs and
watching replays.

**Status 2026-10-24.** The platform is implemented and tested end to end: 22 validated levels, a
25-DoF humanoid, a working PPO trainer with checkpoints and resume, deterministic evaluation,
replay export, a research dashboard, a bounded CEM tuner and a restricted research assistant.
141 automated tests pass and `ruff check src tests` is clean. The *learning* result is partial
and reported honestly in [Learning status](#learning-status): PPO learns to stand and to make
partial progress on the first course, but it does **not** yet complete parkour levels. Details in
[docs/STATUS.md](docs/STATUS.md).

## Quick start

```bash
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"

parkour doctor                 # environment + 22-level catalogue check
parkour list-levels            # difficulty, length, time limit per level
parkour random --level level-01-basics --policy random --steps 400
parkour train --level level-01-basics --run-id demo --steps 200000 --envs 4 --max-minutes 5
parkour evaluate --checkpoint runs/demo/checkpoints/latest.npz --level level-01-basics
parkour dashboard --port 8000  # http://localhost:8000
```

No GPU, display or Node.js is needed. See [docs/TRAINING.md](docs/TRAINING.md) for the full
command reference, reward table and evaluation protocol.

## What works (and is tested)

| Area | State |
|---|---|
| Articulated humanoid | 25 DoF / 19 actuated joints, explicit joint limits, force limits, position and activation-filtered torque actuation, procedural MJCF |
| Courses | 22 fixed manifests validated against a JSON Schema: platforms, steps, stairs, gaps, beams, hurdles, spikes, moving bridges, rooftop sequences, 2 held-out |
| Environment | Gymnasium-shaped reset/step, 90-value documented observation, 19 bounded actions, per-component reward, 8 termination reasons, contact/energy diagnostics |
| Learning | NumPy PPO (GAE + clipped surrogate + obs normalisation), 4 parallel envs, resumable checkpoints, deterministic evaluation, replay export |
| Curriculum | `single`, `uniform` and `adaptive` (EWMA per-level performance, weakness weighting, novelty bonus, periodic retesting of mastered levels) |
| Persistence | `runs/<id>/` with config, provenance (git commit, versions, CPU/GPU), CSV/JSONL metrics, checkpoints, evaluations, replays |
| Dashboard | FastAPI + vendored Three.js: run browser, learning curves, per-level success, replay playback in 3D, run comparison, benchmark, start/stop one training process |
| Secondary optimisers | Bounded CEM over 8 reward/PPO knobs; restricted Harness²-style assistant (11 bounded knobs, proposals only, never applied automatically) |
| Tests | 141 tests: model/geometry, levels, environment mechanics, PPO math, training pipeline, CLI, dashboard endpoints, research tools |

## Learning status

Training is honest about where it is. On `level-01-basics` (a 6 m flat runway with one 10 cm
step), from a randomly initialised policy:

* **Works:** standing and recovering. A posture-only reward learns to stay upright for the whole
  episode (zero action stays up for 2 000+ control steps; a 250 k-step run reaches the episode cap
  in ~56 % of episodes).
* **Partially works:** progress. With an annealed reward (posture → forward), the policy learns to
  move ~2.7 m toward the finish before collapsing, which is 45 % of the first course after 1.7 M
  steps, and it is still improving. Deterministic and stochastic evaluation agree within 0.1 pt.
* **Does not work yet:** completing a course (0 % success, every episode ends in `collapsed`), a
  stable walking gait, and therefore all the harder levels (`level-02-step-up`: 12 % completion).

A one-line bug in the evaluation loop used to report an open-loop controller, which is why older
numbers in this repo looked ~4× worse than the policy actually is. It is fixed, and two tests now
pin the closed loop (`docs/STATUS.md`, §5).

Two dead ends are documented rather than hidden: a dense progress reward from step 0 teaches a
"lunge and fall" policy, and a gait-shaped reward without a no-progress termination teaches
"march in place". A scripted open-loop CPG search (2 000+ parameter combinations) also failed to
produce a stable gait on this morphology, so the remaining gap looks morphological/actuation-relevant
as well as sample-budget-relevant. Concrete next steps are listed in
[docs/STATUS.md](docs/STATUS.md#what-remains-unfinished).

## Architecture in one paragraph

A level manifest plus the humanoid spec compile into one MJCF scene; `ParkourEnv` wraps it as a
Gymnasium-style environment; vectorised envs feed a NumPy PPO learner which writes runs to disk;
evaluation reloads a checkpoint with its recorded dynamics; the dashboard reads the same
directories and replays captured frames in the browser. The physics engine, the controller and the
learning algorithm are never replaced by the optional research tools. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository map

```text
configs/levels/           22 level manifests + level.schema.json
docs/                     ARCHITECTURE, DECISIONS, ROADMAP, RESEARCH, TRAINING, STATUS
src/parkour_lab/
  humanoid.py             articulated humanoid + MJCF generation
  levels.py               manifest loading/validation, catalogue
  course.py               level + humanoid -> MJCF scene
  env.py                  ParkourEnv, rewards, terminations, diagnostics
  policies.py  ppo.py     NumPy actor-critic and PPO update
  training/               vector envs, curriculum, runner, bounded evolution
  evaluation.py           rollouts, aggregation, comparison
  storage.py              run directories, provenance, listing
  dashboard/              FastAPI app + static Three.js front-end
  harness2.py             restricted research assistant
  cli.py                  the `parkour` command
tests/                    nine focused test modules
tools/generate_levels.py  deterministic level authoring (byte-identical regeneration)
```

## Tests

```bash
pytest                 # 141 tests, ~42 s (includes two short real training runs)
ruff check src tests
```

`pytest -m "not slow"` skips the end-to-end training tests. The training tests really train a
small policy, evaluate it, resume it, and compare two runs — they are not mocks.

## Honest scope

* The humanoid is a game/physics model, not an anatomical or biomechanical one. "Muscle" actuation
  means activation-filtered torque with a force ceiling, not Hill-type muscles.
* MJWarp / mjlab were not installed or benchmarked on this 2-vCPU, no-GPU host, so no performance
  claim is made about them; the NumPy PPO baseline is what was measured.
* The dashboard has no authentication and binds to the local machine by default; the training
  endpoint can start one process at a time and is meant for a trusted workstation, not a server.
* Level geometry is authored by hand-tuned generator code; difficulty ordering is by design
  intent, not by measured agent performance.

Training data, checkpoints and replays are ignored by Git (`runs/`).
