# Proposed architecture

The architecture separates **simulation**, **control/learning**, **evaluation**, **persistence**, and **presentation**. There is no implemented runtime application yet; the package contract in `src/parkour_lab/contracts.py` is deliberately limited to the environment/policy/evaluation seams.

## Component map

```text
configs/levels/*.json ──> level loader/validator ──> fixed course definition
                                  │                         │
                                  v                         v
                    environment adapter <──> MuJoCo/MJCF humanoid + course
                                  │ observations / rewards / episode results
                                  v
                    trainer (PPO baseline) <──> policy / optimizer
                                  │                     │
                    optional evolution layer     checkpoint registry
                    (bounded outer-loop search)        │
                                  └──────┬──────────────┘
                                         v
             run directory: config + seeds + metrics + trajectories + models
                       │                                  │
                 TensorBoard                         local API (later)
                                                         │
                                          responsive React/Three.js dashboard

optional researcher/LLM experiment agent ──> restricted candidate config/script
                         └──> fixed offline evaluator ──> human-reviewed run record
```

## Responsibilities and contracts

| Component | Owns | Must not own |
|---|---|---|
| Level data / curriculum | Versioned, fixed course definitions; level order; unlock/evaluation splits | Physics code or hidden changes to held-out levels |
| Physics backend | Humanoid MJCF, joints/actuators, obstacle bodies, contacts, stepping, state extraction | Policy optimization or dashboard state |
| Environment/task adapter | Reset, observation/action mapping, reward, termination/truncation, info and reproducible seeds; Gymnasium-like step contract | Optimizer, checkpoint promotion, web transport |
| Policy and trainer | Policy network/action distribution, PPO rollout/update, optimizer state, training RNG, safe resumable checkpoint | Editing level source or self-grading the held-out suite |
| Evolution adapter (later) | Candidate generation over a declared bounded search space; rollout budget; selection on validation split | Unbounded new processes, model-weight search by default, access to final test scores during tuning |
| Evaluation | Fixed seeded course suite, deterministic/stochastic inference mode, per-level metrics, paired comparisons and aggregate reports | Training updates or changing the test set to improve a score |
| Experiment store | Immutable run ID/config snapshot/revision, event/episode records, checkpoints, replay references, atomic writes | Reliance on cloud services or a dashboard being online |
| Research dashboard (later) | Run browsing, training/evaluation curves, one selected 3D replay/course, launch/stop only through explicit local API controls | Running the physics engine inside the browser or consuming all training GPU memory |
| Harness²-style research agent (optional) | Propose constrained edits/diagnostics; execute candidate experiments through a sandboxed adapter; retain verified lessons | Directly controlling the humanoid, PPO weight training, or promoting a model without approval |

`contracts.py` currently defines only the Gymnasium-shaped environment boundary, a policy inference protocol, and a validated episode summary record. It has no concrete implementation and intentionally avoids importing Gymnasium, NumPy, MuJoCo, PyTorch, FastAPI, or a UI framework in the base package.

## Data flow

1. A level loader validates JSON against `configs/levels/level.schema.json`, converts coordinates and objects into the chosen backend, and rejects an invalid course before a run starts.
2. One environment reset receives a seed and level ID. Observations expose documented proprioception and compact task/obstacle state; actions map through a bounded, explicit actuator interface. The environment reports reward, termination, truncation and diagnostic info separately.
3. PPO samples parallel episodes, stores training state and writes checkpoints and metrics. Every training and evaluation run uses a fresh immutable run directory with copied config and source/backend fingerprints.
4. A separate evaluator loads a checkpoint read-only, traverses the fixed test suite, and writes one episode record per (level, seed). Promotion compares the candidate with a retained baseline on matched seeds; keep score and safety/fall metrics visible.
5. A later API reads run records and supplies aggregated charts and selected state/replay to the browser. Sampling and frame-rate limits prevent visualization traffic from affecting rollout throughput.
6. Optional harness iteration proposes an experiment—not an in-flight policy mutation. Candidate configs run in a fresh process/workspace against the same development/validation evaluation budget; results and diffs are reviewed before any change is promoted. A final held-out split remains unavailable to the proposer.

## Experiment storage layout (planned)

```text
runs/<run-id>/
  config.toml              exact config copy
  provenance.json          code revision, versions, backend, device, seeds
  metrics/episodes.jsonl   per-episode level/seed/success/progress/fall/reward data
  metrics/training.csv    update/throughput/resource curves (or TensorBoard events)
  checkpoints/             policy plus optimizer/RNG for resume; best separately marked
  replays/                  bounded selected episodes; video only when requested
  evaluation/               immutable suite results and comparison report
```

The repository tracks only templates and small test fixtures. Generated data and checkpoint paths are excluded via `.gitignore`. The exact index/store format is deferred until a working run pipeline demonstrates which queries the dashboard needs.

## Key boundaries

- **One MJCF source of truth where possible.** The CPU MuJoCo debug environment and accelerated Warp environment should load the same model/course assets and receive parity checks. They may differ in backend implementation; results must name the backend.
- **Physics vs rendering.** Training simulation is headless and batched. The CPU/native viewer is for early debugging; the eventual browser viewer displays one selected replay/level and is never a trainer dependency.
- **Learning from scratch vs demonstrations.** Random initialization and direct RL are the main starting condition. Motion tracking or AMP is an explicitly separate baseline, not an unnoticed warm start.
- **PPO vs evolution.** PPO is the initial policy optimizer. Evolution is a distinct, budgeted outer experiment interface, initially over compact parameters rather than all network weights.
- **Learner vs Harness².** Neural policy weights and the numeric control objective are learned by RL/EA. Harness² can optionally alter the LLM research agent's harness and inform proposals; it neither simulates nor trains the controller.
