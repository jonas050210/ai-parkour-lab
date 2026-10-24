# Roadmap

Phases are ordered to expose physics, measurement, and compute risks before adding algorithmic or interface complexity. A phase is complete only when its acceptance criteria have evidence in tests or a saved, reproducible run.

## Status as built (2026-10-24)

| Phase | State | Evidence |
|---|---|---|
| 0 — Research and foundation | **done** | docs, schema, starter config, 144 passing tests |
| 1 — One-world physics and measurement | **done** | 25-DoF humanoid, `ParkourEnv`, contact/termination/reward diagnostics, `tests/test_environment.py` (18 tests) |
| 2 — Fixed course suite and reproducible evaluation | **done** | 22 validated levels, deterministic multi-seed evaluation with checkpoint dynamics, `parkour compare`, held-out split |
| 3 — Small-batch PPO baseline | **done, goal not reached** | PPO trains, resumes, checkpoints and evaluates reproducibly; it learns standing and 14–21 % of level-01 but never completes a course. See [STATUS.md](STATUS.md#learning-status) |
| 4 — Bounded evolution / experience | **partial** | CEM over 8 knobs works and is tested (`tests/test_research_tools.py`); no experience-retrieval layer, no PBT comparison |
| 5 — Local 3D research dashboard | **done** | FastAPI + Three.js viewer, run browser, curves, per-level bars, replay playback, comparison, benchmark, validated start/stop; `tests/test_dashboard.py` |
| 6 — Harness²-style assistant | **done (restricted pilot)** | rule-based proposals over 11 bounded knobs, probe reports, no auto-apply; network planner disabled by default |

Phase 3's learning target is the open item: the platform around the learner is complete, but the
humanoid does not yet walk. The concrete diagnosis and the next experiments are in
[STATUS.md](STATUS.md#what-remains-unfinished).


## Phase 0 — Research and foundation (current)

**Scope:** source-checked choices, decision log, package boundary, run/level templates and project plan.

**Acceptance criteria**
- [x] Research, architecture, decisions, roadmap, honest project status are documented.
- [x] Harness² mechanism and non-applicability boundaries are checked against its paper and actual implementation.
- [x] Starter TOML and a versioned JSON level schema/example exist.
- [x] Base Python interfaces avoid pulling in the heavyweight simulation/training/UI stacks.
- [x] Five starter tests, Ruff checks, formatting checks, JSON parsing, and JSON Schema validation pass in the research sandbox.
- [ ] Execute the foundation check in Python 3.12. The sandbox provides Python 3.11; packaging/tests were sanity-checked there with the Python-version metadata bypassed, which does not substitute for validating the declared baseline.

**Exit:** agree the humanoid/modeling fidelity target; confirm the actual computer's OS, driver and GPU memory reporting.

## Phase 1 — One-world physics and measurement

**Dependencies:** Phase 0. **Do not train a learned policy yet.**

**Deliver:** a single articulated humanoid MJCF with meaningful joint limits and collisions; one flat/safe test course and level loader; CPU MuJoCo environment; headless state inspection, seeded reset, stepping, termination and reward instrumentation. Add an optional native viewer for debugging. Begin with a full articulated character under documented position/PD or torque-limited actuators; leave the muscle-actuator path behind a replaceable controller interface.

**Acceptance criteria**
- MJCF compiles reproducibly; body/joint/actuator inventory and masses/limits are checked.
- Reset with the same seed produces the same initial configuration; finite observations/rewards and correct five-value Gymnasium-style transition contract.
- Unit tests exercise stand/fall, contact, course completion, episode timeout (truncation separate from termination), and invalid course rejection.
- A scripted/random policy can be stepped headlessly; at least one recorded trace can be inspected without training.
- Physics timestep, action rate, actuator saturation, reward components, falls and obstacle collisions are logged. No unsupported claim of biomechanical validity.

**Exit:** reward/termination tests are stable; determine per-world cost and acceptable humanoid DOF/contact budget.

## Phase 2 — Fixed course suite and reproducible evaluation

**Dependencies:** Phase 1.

**Deliver:** versioned fixed levels and reset/evaluation tooling. Progress from basic locomotion/low steps to broad jumps/gaps, spikes as hazards, moving platforms, advanced transitions, then height/high-rise challenges. Add course IDs/revisions and a separate held-out course/seed set before tuning.

**Acceptance criteria**
- Each obstacle type has explicit geometry/physics, success condition, failure/timeout behavior and a test.
- Course difficulty is measurable (clearance, gap, timing/speed, landing width and fall cost); level updates require a data revision.
- Paired evaluation runs a fixed baseline on a documented set of seeds, produces per-level success, progress, completion time, falls, and actuator-effort/impact metrics, and does not update policy weights.
- Result records identify course revision, seed, simulator version and source commit; failed/aborted episodes are not silently dropped.

**Exit:** baseline evaluator can compare two policies without access to training code paths.

## Phase 3 — Small-batch PPO baseline

**Dependencies:** Phase 2, an accepted observation/action/reward contract, hardware preflight.

**Deliver:** PPO training with random policy initialization, resumable full-state checkpoints, early easy-level curriculum, local metrics, explicit compute limits. First prove correctness with CPU/single-world or tiny batches; then test the Warp/`mjlab` candidate against the same MJCF task.

**Acceptance criteria**
- One smoke run completes end-to-end and can resume from a checkpoint without resetting optimizer/RNG state silently.
- At least three training seeds are compared on the same frozen validation suite; preserve base, best, and unsuccessful runs.
- Training configs/checkpoints are immutable or versioned; curve logs include environment steps/sec, total steps, peak VRAM/RAM, and backend.
- Starting from a fresh policy is explicitly tested and labeled; any imitation/AMP experiment has its own separate config/result namespace.
- Batch size is increased only after measured peak memory leaves documented headroom on the 8 GB target. No vision training yet.

**Exit:** a statistically interpretable initial result and demonstrated checkpoint recovery exist; no precondition that the agent already completes a parkour level.

## Phase 4 — Controlled hybrid/evolution and experience

**Dependencies:** Phase 3, stable evaluator, per-run data format.

**Deliver:** a bounded optimizer for interpretable search dimensions (first reward/curriculum/PPO hyperparameters or a compact skill/controller latent); archive checkpoint/behavior baselines; optional local experience retrieval. Only then compare PBT, CEM/CMA-ES, or an ERL/CEM-RL direction against PPO alone.

**Acceptance criteria**
- Search space, maximum candidate count, simulator-step budget, seeds, selection rule and validation split are declared before a sweep.
- Every candidate has an immutable config and comparable evaluation; held-out test results cannot drive selection.
- Hybrid results report compute/sample cost as well as success and falls; preserve a PPO-only control group.
- Experience entries cite a measurable run/trajectory and are retained only when the intended transfer is tested.

**Exit:** evidence justifies a specific hybrid method; otherwise retain PPO and drop the additional complexity.

## Phase 5 — Local 3D research dashboard

**Dependencies:** stable Phase 2 evaluation/run schema; Phase 3 useful logs. Can start only after run contracts are stable.

**Deliver:** small local API and responsive React/TypeScript/Three.js viewer, run browser, learning curves, success-by-level comparisons, and replay of one selected episode/course. Launch/stop is explicit and observable. Keep the simulation/trainer in a separate process.

**Acceptance criteria**
- Dashboard panels reflect saved evaluation/training records and agree with CLI reports.
- Browser displays a fixed recorded state/course; absent runs/backend reports an honest empty/error state.
- Static asset/API paths work on the target local setup; no browser code calls its own `localhost` to reach a separate service.
- Rendering one replay does not pause the headless trainer or change its policy/environment state.
- Responsive layout, keyboard-accessible controls and bounded update frequency are verified.

**Exit:** user can inspect and compare real saved runs; mock metrics are clearly labeled if ever used.

## Phase 6 — Harness²-style research-assistant experiment (optional)

**Dependencies:** at least a stable Phase 2 evaluator and restricted experiment interface. No need to adopt upstream Harness² just to build the RL system.

**Deliver:** one tiny, human-reviewed Propose–Probe–Compose pilot over a narrow, versioned edit surface (for example reward weights or a training YAML/JSON/TOML config) or a constrained diagnostic harness. Run candidates on identical development seeds; write evidence and changes to separate run directories. Keep final test levels inaccessible.

**Acceptance criteria**
- The adapter, chosen LLM editor/executor, costs, time/compute budget, sandbox, network policy and editable files are explicit.
- Every candidate is validated against schema/ranges before execution; unreviewed scripts cannot execute with broad host/network permissions.
- A baseline, matched candidate probes, multi-seed evaluator, and no-edit/failed-edit outcomes are all recorded.
- Retained experience points to results and survives a later held-out replication; a tool choosing a winner once is not proof of general improvement.
- Policy weights are never changed by Harness² unless the separate, explicit RL/EA trainer is invoked under the run budget.

**Exit:** documented evidence shows whether the research-assistant harness is useful; integration remains optional and can be rejected.

## Immediate next steps

1. Resolve the model/actuation and simulation operating-system choices in [DECISIONS.md](DECISIONS.md).
2. Check the target host's CUDA driver, Python 3.12, available VRAM and disk/RAM before installing `mjlab`/Warp.
3. Implement Phase 1's single-world model and tests; do not start a population or long training job.
