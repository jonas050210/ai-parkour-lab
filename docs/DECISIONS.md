# Decision log

**Status date:** 2026-10-23. Decisions are recommendations for the next implementation phase, not proof of running code. Revisit with evidence when the first humanoid and target-machine profile exist.

## Accepted for the foundation

### D-001 — Keep the base package dependency-free; Python 3.12 is the project target

- **Decision:** use Python 3.12 as the supported starting environment; keep simulation, training and web/UI dependencies optional.
- **Context:** MuJoCo itself can be used from Python, while the current `mjlab` stack includes Warp, PyTorch, MuJoCo and RSL-RL and has CUDA-specific packaging considerations. A GUI stack is another independent dependency set. Pulling these all into a basic import would make documentation/testing require the full GPU environment.
- **Consequences:** interfaces and data templates can be checked on CPU without installing physics or AI libraries. Python 3.12 should be validated against the target OS/driver before simulator installation. `pyproject.toml` extras are later integration points, not working features.

### D-002 — MuJoCo/MJCF is the candidate source of truth; keep a single-world CPU reference

- **Decision:** author model/course assets in MJCF, first validate them with ordinary MuJoCo in one CPU world; retain a backend boundary for later batched GPU execution.
- **Evidence:** official MuJoCo provides an articulated-body model format and actuators; official MJWarp targets NVIDIA throughput and is explicitly complementary to single-world low-latency MuJoCo. `mjlab` composes MJWarp with modular task definitions and PyTorch.
- **Trade-off:** Warp batching and the `mjlab` stack are a stronger route for local accelerator sampling, but actual GPU fit and high-DoF performance are unknown on 8 GB. A CPU reference provides a simpler debugging and fallback route.
- **Revisit if:** target host cannot support the pinned CUDA stack, an implementation shows problematic Warp performance/feature support, or a measured alternative clearly fits better.

### D-003 — PPO is the first policy optimizer; evolutionary search follows a baseline

- **Decision:** measure random initialization and PPO first. Add an outer-loop, budgeted evolutionary comparison over compact reward/curriculum/hyperparameter or skill-latent variables after baseline evaluation is reproducible.
- **Evidence:** PPO is the current robotics-first path in RSL-RL and integrated in `mjlab`; it makes an inspectable on-policy baseline. CEM/CMA-ES/PBT and ERL/CEM-RL exist as research alternatives, but high-dimensional weight evolution multiplies costly rollout requirements and may be noisy on the available hardware.
- **Trade-off:** this stages the requested hybrid learning rather than claiming a hybrid trainer exists. “Experience-based improvement” first means retaining checkpoints, trajectory/run evidence and tested lessons; a replay/retrieval algorithm is a separate later decision.
- **Revisit if:** PPO is demonstrably unstable or sample-inefficient on the minimal task and a controlled alternative has a clear evaluation advantage.

### D-004 — Start with an articulated full body and bounded joint-level actuation; separate the muscle-fidelity question

- **Decision:** retain a humanoid's important body segments, joints, collisions and limits; start with documented joint-target/PD or torque-bounded actuation. Define an adapter seam for an activation-driven muscle-actuator experiment later.
- **Context:** the project requests muscle-like actuation, but high anatomical fidelity (muscle paths, parameters and a validated musculoskeletal body) is different from merely simulating a humanoid with joint motors. MuJoCo/mjlab expose muscle actuator types, but that does not supply a validated full human model automatically. A large muscle model may also change Warp performance materially.
- **Trade-off:** the initial controller is muscle-inspired only in bounded force/control realism, not an anatomical muscle simulation. A later side-by-side activation-actuator comparison can test what added fidelity contributes.

### D-005 — Fixed versioned level manifests and held-out evaluation

- **Decision:** describe courses as version-controlled JSON manifests validated by a JSON Schema; separate training/curriculum levels from held-out evaluation. Keep the schema/example physics-neutral.
- **Evidence:** fixed levels enable matched-seed comparisons; a physics-neutral manifest can later be loaded by both CPU MuJoCo and an accelerated backend. A schema does not implement course geometry.
- **Consequence:** schema or level edits get explicit revisions and need review; the future loader must add physics validity checks beyond JSON schema validation.

### D-006 — Local run directories and TensorBoard first; dashboard later

- **Decision:** persist local immutable run folders, exact configs/provenance, portable JSONL episode records and checkpoints; add TensorBoard for curves. Defer a database/API/web app until those records stabilize.
- **Evidence:** this is small, testable and supports offline work. React + Three.js/React Three Fiber is a suitable 3D UI path, but displaying the viewer is not a prerequisite to training.
- **Trade-off:** file queries/aggregation may need a small SQLite index later. Cloud experiment tracking is optional and is not part of the local-first baseline.

### D-007 — Harness² is an optional developer-facing experiment method, not the controller

- **Decision:** do not integrate Harness² into the simulator or PPO loop. If explored later, adapt its evidence-based proposal/probe/composition pattern to a restricted, human-reviewed experiment assistant that proposes config or diagnostic changes.
- **Evidence:** its official source edits eight components of an LLM execution harness around frozen model weights and ships OpenCode/Codex executors plus JobBench/LAB/WorkBuddy adapters. It supplies neither physical simulation nor numeric-policy learning nor a parkour adapter. The upstream repository says it is not an officially supported Google product.
- **Safety/research condition:** config schema/range validation, sandbox/egress restrictions, separate workspaces, matched seeds, fixed held-out levels, logged cost, human checkpoint approval. No automatic run promotion or model mutation.

## Open decisions / questions for the next design review

These are the only questions that need a product/owner preference; they are not prerequisites for writing the Phase 1 prototype. Recommendations below are current best judgments.

### Q1 — What does “muscle-like” mean for the first milestone?

- **Context/evidence:** MuJoCo/mjlab expose an activation-driven muscle actuator, but anatomical realism also requires a chosen validated musculoskeletal body and parameters. More actuators/flexibility raise model and training complexity; Warp documentation flags performance work for a single connected system above about 60 DoFs.
- **Alternatives:** (A) full articulated humanoid with bounded joint PD/torque first, then swap/compare muscle-actuator groups; (B) begin with a detailed, anatomically muscle-actuated body and accept the asset/compute burden.
- **Recommendation:** (A) for the first measurable learning baseline, while retaining muscle activation as an explicitly planned experiment. If anatomical muscle realism is central to the research claim, choose B before implementation and narrow scope elsewhere.

### Q2 — What host OS and simulation identity are intended?

- **Context/evidence:** Python 3.12, NVIDIA driver/CUDA compatibility and Warp wheel availability govern local installation. No user-machine probe was possible in this research phase, and CPU Python 3.11 in the research sandbox is not evidence of the target GPU host.
- **Alternatives:** native Linux; Windows with a deliberately verified native environment; Windows host + Linux/WSL2 for GPU simulation.
- **Recommendation:** Linux (or verified GPU-enabled WSL2) for `mjlab`/MJWarp training; retain a regular CPU MuJoCo path for portability. Confirm the actual system, driver, and whether the RTX 4060 Ti is the 8 GB model before pinning/installing.

### Q3 — Is the dashboard a must-have in the first application milestone?

- **Context/evidence:** 3D viewing and research plots are a desired outcome, but the physics/task/run schema are currently unspecified. Building the UI first would create mock state and duplicated contracts. Rendering also competes for resources during training.
- **Alternatives:** (A) native viewer and CLI first, build a 3D web dashboard once records stabilize; (B) create the web app/API shell before training.
- **Recommendation:** (A), then a local React/Three.js viewer consuming saved runs. Choose B only if early user evaluation is a project constraint and mock data is clearly separated from real training records.

## Deferred decisions

- Specific humanoid asset/morphology, segment masses, geometry, joint ranges and source/license.
- Final actuator model (motor, PD, Hill-type muscle), action rate, observation set, reward values and whether the agent sees obstacle geometry or pixels.
- Exact CPU/GPU backend, version lock, CUDA wheels, world batch size and performance target after target-machine profiling.
- PPO network shape, normalization, curriculum unlock policy, randomization, evolutionary method/search range and statistical test plan detail.
- Whether experience storage should use SQLite, MLflow, or raw files only; API transport and dashboard implementation.
- If Harness² is used: model/editor/provider, executor substrate, sandbox, adapter design, cost ceiling, policy and network-egress settings. These cannot be answered from its paper alone for a local parkour project.
