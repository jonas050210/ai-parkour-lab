# Research findings

**Research snapshot:** 2026-10-23. Primary project repositories, current documentation, paper, and package metadata were checked. External performance numbers are quoted only when they are a primary-source claim; no throughput figure is assumed to apply to this humanoid or the RTX 4060 Ti 8 GB.

## Executive conclusion

Build a small, inspectable simulation and evaluation loop before attempting a hybrid learner or dashboard. Use **MuJoCo/MJCF** as the shared model/course format, make a single-world CPU environment the physics and reward reference, and test **MuJoCo Warp via `mjlab`** as a separate batched NVIDIA training implementation. Begin learning with **PPO** and a full articulated body under bounded joint-position/PD or torque actuation. Add evolutionary search as a measured outer loop (for low-dimensional reward, curriculum, or training hyperparameters) only after a seeded PPO baseline exists. Store runs locally and make each result traceable to its config, level revision, seed, code revision, and checkpoint.

This choice is a research starting point, not a claim that the complete stack has already been installed, fits into 8 GB, or achieves a particular sample rate. The target machine must be benchmarked with the actual humanoid, contacts, observation sizes, and training batch. Start headless, keep the interactive viewer separate, and scale environments only after measuring peak VRAM and system RAM.

## Harness²: what it actually does and where it fits

**Primary sources:** the [Harness² paper (arXiv:2610.14224)](https://arxiv.org/abs/2610.14224), the [official `google-research/harness2` repository](https://github.com/google-research/harness2), and its [`core/recursion.py`](https://github.com/google-research/harness2/blob/main/harness2/core/recursion.py), [`core/spec.py`](https://github.com/google-research/harness2/blob/main/harness2/core/spec.py), [`core/ledger.py`](https://github.com/google-research/harness2/blob/main/harness2/core/ledger.py), [`core/library.py`](https://github.com/google-research/harness2/blob/main/harness2/core/library.py), and [experiment guide](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md). The source was cloned and inspected at commit `1cfe254e8d25b15f37048c3730134686abe60335`; its five repository workflow tests passed in this research environment.

Harness² improves the **execution harness around a fixed-weight LLM**, not an embodied agent's neural policy and not the physical simulation. Its public implementation has two recursions:

1. **Task-level Propose–Probe–Compose:** run a reference execution; propose general and task/domain-specific harness edits; execute candidate harnesses on the same task; compose a successor from the contrasts. It supports parallel exploration and sequential refinement.
2. **Harnessing-level experience:** reflect on the complete round, retain adopted harnesses and evidence-supported lessons in a persistent improvement library, and use that library to guide later proposals. The next task starts with the common base execution harness; earlier harnesses are selectively reused rather than blindly inherited.

The code's harness schema covers eight LLM runtime components: system prompt, guardrails, memory, tool guidance, subagents, skills, scripts, and plugins. Its shipped executors are OpenCode and Codex, and its benchmark adapters target JobBench, LAB, and WorkBuddy-Bench. It is **not** a MuJoCo engine, humanoid model, RL algorithm, evolutionary optimizer, checkpoint manager for policy weights, or ready-made parkour adapter. Its README also warns that it is not an officially supported Google product. The paper's benchmark gains are author-reported results on professional-work tasks and should not be transferred to parkour learning.

**Credible project application:** optionally use Harness² or its research pattern to improve the *developer-facing experiment assistant*—for example, propose a reward/curriculum experiment plan, edit a constrained experiment config or diagnostic script, run the fixed evaluation suite, compare trajectory evidence, and retain reusable findings. A parkour-specific adapter/executor, task schema, safe edit surface, and verifier are additional work. Keep the actual policy optimization under PPO/evolutionary code, never allow an LLM to alter an active training run or promote a checkpoint without explicit evaluation and approval. Use held-out fixed levels and multiple seeds; same-task behavioral contrast alone can be stochastic and is not proof of generalization. Candidate configs/code should be versioned, bounded, and reviewed. Do not wire untrusted model-generated scripts into an unrestricted executor.

The direct paper mechanism is about editing an LLM harness and does not promise measurable improvement in a numeric control policy. Applying its *experimental discipline*—controlled proposals, matched probes, evidence-based retention—is a design analogy, not a Harness² capability.

## Physics, humanoid and actuator choices

### MuJoCo (recommended physics foundation)

[MuJoCo](https://github.com/google-deepmind/mujoco) is a fast, open-source articulated-body engine. Its official [MJCF modeling guide](https://mujoco.readthedocs.io/en/latest/modeling.html) describes a hierarchical model format, joints, actuators, contacts and incremental model refinement. The [XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html) includes the native actuator types, including muscle actuators. MJCF is a good common artifact for fixed courses, agent evaluation, and the later batched backend. Start with native rendering/viewer and one physics world, not photoreal rendering.

Do not conflate a MuJoCo muscle actuator with a validated anatomical musculoskeletal avatar. Muscle actuators add activation/force dynamics, but body geometry, muscle paths, strength, contact and physiological validity still need to be selected and checked. For the initial policy, a complete articulated humanoid driven by bounded joint targets with PD/torque limits is easier to debug. Keep the actuator interface replaceable so activation-driven muscles can be compared after basic contact, fall and jump behavior works. This is a fidelity trade-off to document, not an assumption that torque motors are literal muscles.

### Candidate batched stack: `mjlab` + MuJoCo Warp

The [MuJoCo Warp manual](https://mujoco.readthedocs.io/en/latest/mjwarp/) says MJWarp is a Warp implementation jointly maintained by NVIDIA and Google DeepMind, targets GPU batched throughput, and works with PyTorch. It distinguishes throughput from single-world latency, and cautions that a single connected mechanism beyond roughly 60 degrees of freedom remains an active performance challenge. That caution matters for high-DoF humanoids and dense muscle models. MuJoCo's [MJX comparison](https://mujoco.readthedocs.io/en/latest/mjx.html) is worth revisiting if JAX transformations or differentiable dynamics become a priority; Warp currently does not offer MJX-JAX-style automatic differentiation.

[`mjlab`](https://github.com/mujocolab/mjlab) combines MuJoCo Warp with a composable, Isaac-Lab-inspired environment API, a PyTorch-native interface, RSL-RL/PPO integration, and built-in motor, PD, and muscle actuator types ([official docs](https://mujocolab.github.io/mjlab/main/index.html), [actuator docs](https://mujocolab.github.io/mjlab/main/source/actuators.html)). Its current project metadata at the research snapshot reports version `1.6.0`, Python `>=3.10,<3.15`, PyTorch `>=2.14.0`, Warp, MuJoCo/MuJoCo-Warp and RSL-RL dependencies. The repository was inspected at `8bb2bc9e187797689ec82c78c35019afd2fd0376`. This is a promising fit, but the full current dependency stack and CUDA wheels are a notable setup/VRAM risk on a 4060 Ti with only 8 GB. Do a small install and backend parity test; do not make it a mandatory base dependency yet.

### Alternatives

| Option | Strengths | Limitations for this project | Recommendation |
|---|---|---|---|
| MuJoCo CPU + `mujoco.rollout` | Small entry point; good single-world debugging and interactive latency; shared MJCF | Batching is CPU-side; CPU simulation to GPU learning can incur transfer costs | Required reference path and fallback |
| MJX-JAX | JAX `vmap`/`pmap`; supports JAX transformations and differentiable dynamics; can target more than NVIDIA hardware | Different software stack from PyTorch; workload/model feature and memory behavior need validation | Keep as a future comparison if differentiation or JAX is compelling |
| MuJoCo Warp (`mjlab`) | NVIDIA GPU batch simulation, PyTorch-centric policy stack; manager-style observations/rewards/curriculum; actuator choices include muscle | GPU and system memory contention; CUDA setup; Warp connected-DoF scaling caveat; version churn | First accelerated candidate, benchmark actual task before adoption |
| Isaac Lab / Isaac Sim | Mature robotics ecosystem, GPU parallel simulation, rich rendering/sensors, reusable humanoid assets/tasks | Much larger omniverse/runtime footprint and more complex install; high rendering/VRAM contention; unnecessary for state-based starter task | Revisit if realistic vision/sensors or a compatible ready-made robot task becomes essential |
| Brax | JAX-native fast batched physics/learning | Separate dynamics/contact and model ecosystem from the chosen MuJoCo MJCF reference; extra backend parity work | Not selected as the first backend |
| PyBullet | Accessible API and broad examples | No comparable selected GPU-batched path for this workload; adds another physics implementation to validate | Not selected |

**Hardware guidance, not a performance promise:** keep the initial course state-based (proprioception plus compact obstacle/goal features). Train headless. Start at a deliberately small batch (e.g. 64 worlds), record peak memory, and scale in controlled increments; the exact usable batch is unknown until the robot model exists. Save headroom for the OS, CUDA context, optimizer, and other applications. Avoid rendering every training world or running the dashboard's 3D viewport in the same GPU process. CPU MuJoCo is a useful proof-of-correctness path even if it trains more slowly.

## Parkour-specific evidence

Two recent primary-source humanoid parkour papers help distinguish useful future extensions from the project's deliberately small first task:

- [Deep Whole-body Parkour](https://arxiv.org/abs/2601.07701) combines exteroceptive depth with whole-body motion tracking. Its pipeline aligns human demonstrations and terrain, then uses reinforcement learning to reproduce/adapt agile motions. The paper motivates whole-body contacts such as vaulting and dive-rolling, but this is not a clean “random policy learns every movement without prior skill” baseline.
- [Hiking in the Wild](https://arxiv.org/abs/2601.07718) describes single-stage reinforcement learning from depth input and proprioception to joint actions, plus geometric foothold-safety and target-sampling methods; the paper also uses AMP-style reward terms. It highlights perception, edge safety, and reward-hacking risks, but its unstructured-terrain / hardware-transfer scope is considerably larger than fixed, state-based starter levels.

These papers support a staged plan: begin with compact state/obstacle features on fixed deterministic courses, where physics and reward can be understood; add camera/depth sensing, hand contacts, or motion-prior comparisons only after the low-dimensional baseline is measurable. Record a motion-tracking/AMP result as a separate baseline if used, rather than describing it as training without prior movement data.

## Learning strategy

### Reinforcement learning baseline

Use **PPO** first, with an explicit continuous action definition, clipped actuator targets, a strong fall/effort/impact termination-and-reward design, fixed level reset logic, and evaluation outside the training loop. PPO is on-policy and can require many fresh simulated samples, but is a well-established, direct baseline for batched locomotion; it lets physics, reward, and reset bugs be distinguished before adding replay or population machinery. The `mjlab` framework integrates with [RSL-RL](https://github.com/leggedrobotics/rsl_rl), a robotics-focused PyTorch library whose current documentation/research centers on PPO/behavioral cloning and TensorBoard logging.

Do not seed the first result with a parkour demonstration if the central claim is learning parkour from scratch. Evaluate optional motion imitation or Adversarial Motion Priors (AMP) later as an explicitly labeled ablation/transfer baseline; these can improve style or skills but change the learning-from-zero question. A hierarchical/skill policy is a later experiment once separate behaviors are measurable.

### Evolution and experience

Do not combine every algorithm in the first training run. After a reproducible PPO baseline:

- Use **CEM/CMA-ES or a compact population-based method** over lower-dimensional, interpretable choices (reward weights, curriculum thresholds, PPO hyperparameters, or a low-dimensional skill/controller latent), with a fixed evaluation budget and held-out validation. Avoid black-box evolution of every neural weight as the first method: expensive full rollouts and noisy return estimates make it a poor fit for limited local compute.
- Consider population-based training (PBT) if the measured need is concurrent hyperparameter adaptation, or ERL/CEM-RL as a later algorithm comparison. These are more machinery and should earn their place with a small benchmark.
- Implement experience first as **replayable artifacts and structured run summaries**: level/seed, success, furthest checkpoint, falls, progress, time, impact/effort metrics, and a trajectory/replay reference. The first policy can use checkpoint/skill snapshots as frozen baselines. An off-policy replay buffer or skill retrieval mechanism is an algorithm decision for a later phase, not implied by Harness²'s LLM experience library.

For stochastic runs, compare matched seeds/configs, report mean and spread (preferably confidence intervals once the sample count supports them), success rate by level, and out-of-distribution held-out courses. Preserve unsuccessful outcomes.

## Storage, tracking and reproducibility

Use simple local storage first: immutable per-run directory with config snapshot, source revision, dependency/backend metadata, seeds, episode JSONL/CSV metrics, TensorBoard event files, video/replay pointers, and checkpoint(s) containing policy **and** optimizer/training state when resumption is intended. Write files atomically and never let a process silently overwrite a different experiment. Keep generated files under ignored `runs/` or `data/`; do not commit weights, datasets, or high-volume media. A small SQLite index can power dashboard queries later; it must not replace the raw, portable run records.

**TensorBoard** is the simplest first local curve viewer and integrates with the likely PPO stack. [MLflow](https://mlflow.org/docs/latest/) is a possible later local-first experiment registry when filtering, artifact references, and run comparisons outgrow files. Hosted W&B is optional, not required; avoid making network service or credentials part of a local-first prototype. Determinism is limited by GPU kernels and simulator versions, so record the exact seed/backend and measure variation rather than promising bitwise reproducibility.

## Dashboard and 3D client

A small **React + TypeScript + Vite** client with **Three.js through React Three Fiber** is a suitable responsive viewer/dashboard: the official [R3F docs](https://r3f.docs.pmnd.rs/getting-started/introduction) position it as a React renderer for Three.js, making 3D scene composition compatible with ordinary responsive UI. Pair it later with a local **FastAPI** service for experiment/run APIs and file-backed/SQLite metadata. Keep the dashboard optional and API-driven. Send sampled state, metrics and replay/video data to the browser; do not stream simulator internals or place Python physics behind browser-localhost assumptions. Optimize for one displayed/replayed environment while batch training runs headlessly.

Alternatives include a native MuJoCo viewer for the early physics-debug loop (less web/dashboard work) and a small custom Three.js viewer with no React abstraction (less UI dependency, more manual scene integration). Begin with the MuJoCo viewer; build the web command center after training and run records have stable contracts. No UI framework, API, streaming protocol, or web dependency is implemented in this phase.

## Sources

Primary and project sources consulted (accessed 2026-10-23):

1. [Harness² paper, Xu et al., arXiv:2610.14224](https://arxiv.org/abs/2610.14224) and [official repository](https://github.com/google-research/harness2) (implementation commit listed above).
2. [Harness² README](https://github.com/google-research/harness2/blob/main/README.md), [setup](https://github.com/google-research/harness2/blob/main/docs/SETUP.md), [experiments](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md), and source modules linked in the Harness² section.
3. [MuJoCo source](https://github.com/google-deepmind/mujoco), [MJCF modeling](https://mujoco.readthedocs.io/en/latest/modeling.html), [XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html), [MJX](https://mujoco.readthedocs.io/en/latest/mjx.html), and [MuJoCo Warp guide](https://mujoco.readthedocs.io/en/latest/mjwarp/).
4. [`mujocolab/mjlab`](https://github.com/mujocolab/mjlab), [documentation](https://mujocolab.github.io/mjlab/main/index.html), [actuators](https://mujocolab.github.io/mjlab/main/source/actuators.html), [training with RSL-RL](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html), and [mjlab paper](https://arxiv.org/abs/2601.22074).
5. [RSL-RL source](https://github.com/leggedrobotics/rsl_rl) and [paper](https://arxiv.org/abs/2509.10771).
6. [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab) and [Brax repository](https://github.com/google/brax) for alternatives.
7. [React Three Fiber introduction](https://r3f.docs.pmnd.rs/getting-started/introduction), [Three.js](https://threejs.org/), and [FastAPI](https://fastapi.tiangolo.com/).
8. [Torch checkpointing guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [TensorBoard](https://www.tensorflow.org/tensorboard), and [MLflow tracking](https://mlflow.org/docs/latest/).
9. [EvoTorch source](https://github.com/nnaisense/evotorch), [evosax source](https://github.com/RobertTLange/evosax), and [EvoRL paper](https://arxiv.org/abs/2501.15129) for evolution strategy and hybrid algorithm candidates.
10. [Deep Whole-body Parkour](https://arxiv.org/abs/2601.07701) and [Hiking in the Wild](https://arxiv.org/abs/2601.07718) for parkour-related whole-body motion tracking, exteroception, foothold safety and curriculum/reward design.

Papers and repositories can be newer or change after this snapshot. Pin dependencies and record the exact commit/release when an implementation phase begins. Secondary simulator benchmark claims found during research were not used as a sizing promise.
