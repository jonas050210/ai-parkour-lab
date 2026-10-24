# Decision log

**Status date:** 2026-10-24. D-001 … D-007 were written before any code existed; they are kept
for provenance and marked where implementation superseded them. D-008 … D-015 are decisions
taken while building and measuring the working system, each with the evidence that drove it.

## Accepted for the foundation

### D-001 — Keep the base package dependency-free; Python 3.12 is the project target

- **Decision:** use Python 3.12 as the supported starting environment; keep simulation, training and web/UI dependencies optional.
- **Context:** MuJoCo itself can be used from Python, while the current `mjlab` stack includes Warp, PyTorch, MuJoCo and RSL-RL and has CUDA-specific packaging considerations. A GUI stack is another independent dependency set. Pulling these all into a basic import would make documentation/testing require the full GPU environment.
- **Consequences:** interfaces and data templates can be checked on CPU without installing physics or AI libraries. Python 3.12 should be validated against the target OS/driver before simulator installation. `pyproject.toml` extras are later integration points, not working features.
- **Superseded in part (D-013):** the implemented package requires NumPy + MuJoCo, the dashboard requires FastAPI/uvicorn, and the supported interpreter is 3.11–3.13 (`requires-python = ">=3.11,<3.14"`); the only development interpreter actually exercised here is 3.11.2.

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
- **Superseded (D-014):** run directories landed as described, but the viewing layer is the local FastAPI + Three.js dashboard in `src/parkour_lab/dashboard`, and TensorBoard was not adopted.

### D-007 — Harness² is an optional developer-facing experiment method, not the controller

- **Decision:** do not integrate Harness² into the simulator or PPO loop. If explored later, adapt its evidence-based proposal/probe/composition pattern to a restricted, human-reviewed experiment assistant that proposes config or diagnostic changes.
- **Evidence:** its official source edits eight components of an LLM execution harness around frozen model weights and ships OpenCode/Codex executors plus JobBench/LAB/WorkBuddy adapters. It supplies neither physical simulation nor numeric-policy learning nor a parkour adapter. The upstream repository says it is not an officially supported Google product.
- **Safety/research condition:** config schema/range validation, sandbox/egress restrictions, separate workspaces, matched seeds, fixed held-out levels, logged cost, human checkpoint approval. No automatic run promotion or model mutation.

## Decisions taken during implementation (evidence-based)

### D-008 — Radians at the model boundary, degrees only in manifests, limits and reports

- **Decision:** every value that touches `mjModel`/`mjData` (qpos, ctrl, joint targets, limits) is
  in radians. Degree values exist only in level manifests, `EnvConfig.reset_joint_noise_degrees`
  and human-readable tables, and are converted exactly once, at the boundary.
- **Evidence:** the first action mapping multiplied a normalised action by a *degree* joint span
  and then wrote the result into a radian `ctrl` field. Actions were amplified by roughly 57×, so
  the humanoid exploded on contact and every early stability measurement was an artifact.
- **Consequences:** `humanoid.JointSpec` keeps its `lower`/`upper`/`default` fields in degrees for
  authoring, `ParkourEnv.joint_limits_rad` is the single derived array used by control, and a
  regression test asserts the action span equals `(upper−lower)·0.5·action_scale` in radians.

### D-009 — One control rate: 20 Hz default, 50 Hz available, chosen by measurement

- **Decision:** `physics_timestep=0.004` with `control_decimation=5` (20 Hz control) is the default
  for training; 0.002 × 5 (50 Hz) remains available via `--physics-timestep`.
- **Evidence:** on the 2-vCPU host the same policy code runs at ~1.8–2.2 k transitions/s at 20 Hz
  and ~0.8–1.2 k/s at 50 Hz, and the 20 Hz configuration is the one that reliably learns to stand
  (zero action stays up for the full episode; a posture-only PPO run reaches the 6 s cap in ~56 %
  of episodes after 250 k steps).
- **Consequences:** all recorded runs store `physics_timestep`, `control_decimation` and the derived
  `control_dt`, and evaluation/replay rebuild them from the checkpoint (see D-011).

### D-010 — Fixed, documented observation and action layouts with explicit smoothing

- **Decision:** 90 observations (33 task + 19 joint positions + 19 joint velocities + previous
  action) and 19 actions in `[-1, 1]` around the standing pose, with a first-order action filter
  (`action_smoothing`, default 0.6).
- **Evidence:** an undocumented observation block made the first debugging pass guesswork; a
  regression test now checks that the root/goal/task/proprio blocks never overwrite each other.
  Noise sweeps (σ = 0.05 … 0.5) show white-noise actions destroy the robot within 250-700 control
  steps, and smoothing makes the exploration signal usable without changing the optimum.
- **Consequences:** `OBSERVATION_NAMES`, `OBS_JOINT_POS_START` and `ACTION_DIM` are exported and
  covered by tests; changing the layout requires bumping `PARAM_FORMAT_VERSION`.

### D-011 — Evaluation, replay and probes must reuse the checkpoint's dynamics and randomise resets

- **Decision:** `env_config_from_checkpoint()` rebuilds the recorded `action_scale`,
  `action_smoothing`, timestep, decimation, actuation and reward; `rollout_episode(..., randomize_reset=True)`
  perturbs the initial state per seed while keeping the action deterministic.
- **Evidence:** two bugs found by comparing training and evaluation behaviour. (1) `rollout_episode`
  rebuilt `EnvConfig` field-by-field when recording frames, silently dropping action scaling,
  smoothing, actuation and reward — replays showed different dynamics from training. (2) With a
  deterministic policy *and* disabled reset noise, every evaluation seed played the identical
  episode, so a 3-seed evaluation was really one sample and looked artificially stable.
- **Consequences:** the CLI `evaluate`, `replay`, the runner's periodic evaluation and the
  Harness² probe all go through the same helper; seeds are still exactly reproducible because the
  reset noise is drawn from a per-seed generator.

### D-012 — Reward annealing plus a gait term instead of one hand-tuned dense reward

- **Decision:** training may start from a posture-only reward and linearly anneal to the final
  weights (`--reward-ramp-fraction`), and the reward includes an optional gait term that credits a
  clean left/right foot exchange, gated on forward speed.
- **Evidence:** with a dense forward-progress reward from step 0 the policy reliably discovers
  "lunge forward and fall" (returns rise, completion plateaus at 14–16 % of the course, every
  episode ends in `collapsed`/`fell_over` after ~2.5 s). A posture-only reward, by contrast, learns
  to stand. The gait term exists because a pure velocity reward has a stand-still local optimum
  and a dive-forward local optimum with nothing in between.
- **Consequences:** both terms default to off (`reward_start=None`, `gait_per_second=0`) so existing
  runs keep their exact reward; a ramp and its start weights are recorded in `config.json` and
  restored by `TrainConfig.from_dict`.

### D-013 — NumPy-only PPO; MJWarp/mjlab is an unverified alternative, not a dependency

- **Decision:** the learner (`policies.py`, `ppo.py`) is NumPy-only, single-process, with optional
  subprocess env workers. MJWarp/mjlab are not installed or used.
- **Evidence:** the available host has 2 vCPUs, ~3 GB RAM, no GPU, no CUDA, and no OpenGL. A
  measured benchmark (`parkour benchmark --envs 1,2,4 --workers 1,2`) gives ~3.2–3.4 k steps/s for
  one environment and ~4.1 k/s for four serial environments; threads scale at 0.5× (GIL) and
  subprocess workers only win above roughly six environments, so the defaults are `num_envs=4,
  workers=1`. No MJWarp measurement was possible here, so no performance claim is made.
- **Consequences:** the project runs end-to-end on CPU; a GPU backend stays a seam
  (`VectorEnv` protocol) rather than a promise.

### D-014 — Dashboard is server-rendered HTML plus vendored Three.js, no build step

- **Decision:** `parkour dashboard` serves a FastAPI app with static HTML/CSS/ES modules and a
  vendored `three.module.js`; the browser never simulates physics and never talks to a second
  service.
- **Evidence:** the target machine has no GPU/OpenGL, so a native viewer is unusable there, while a
  browser viewer works from a saved replay file. A build toolchain (React/Vite) would add a
  Node dependency for a single-page viewer; the vanilla implementation is ~1 000 lines and is
  covered by endpoint tests. D-006's TensorBoard suggestion is superseded: the dashboard reads the
  run directories directly.
- **Consequences:** no npm install is needed to use the dashboard; `three.module.js` is tracked in
  the repository and updates are a manual vendoring step.

### D-015 — Secondary optimisers are bounded and never auto-applied

- **Decision:** the CEM evolution layer may only vary eight declared parameters inside hard
  bounds, always by training a real PPO run; the Harness² assistant may only propose changes to
  eleven bounded knobs, rejects anything under `humanoid|level|course|physics|observation|
  action_mode|actuation|policy|network|checkpoint`, clamps out-of-range values, and never writes
  to a run or promotes a checkpoint.
- **Evidence:** both rules are enforced by tests (`tests/test_research_tools.py`), including
  rejection of geometry/physics edits and clamping of absurd values.
- **Consequences:** neither tool can silently replace the physics engine, the controller or the
  learning algorithm, which is the standing project constraint.

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
