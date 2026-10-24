# Training, evaluation and replay guide

Everything below runs from the repository root with the project installed
(`python -m pip install -e ".[dev]"`). The commands are the ones used to produce the numbers in
[STATUS.md](STATUS.md).

## Quick recipes

```bash
# 1. Sanity-check the machine (versions, CPU, MuJoCo, level catalogue)
parkour doctor

# 2. Watch a random policy fall over (headless, no display needed)
parkour random --level level-01-basics --policy random --steps 400

# 3. Train a short run and look at the artefacts
parkour train --level level-01-basics --run-id demo --steps 200000 --envs 4 \
  --physics-timestep 0.004 --decimation 5 --eval-interval 20 --max-minutes 5

# 4. Evaluate the checkpoint deterministically on the level
parkour evaluate --checkpoint runs/demo/checkpoints/latest.npz --level level-01-basics \
  --eval-seeds 11 23 37

# 5. Re-simulate one episode into a replay the dashboard can play
parkour replay --checkpoint runs/demo/checkpoints/latest.npz --level level-01-basics \
  --seed 11 --out runs/demo/replays/manual-seed11.json.gz

# 6. Open the dashboard (serves on 0.0.0.0:8000)
parkour dashboard --port 8000
```

On a headless host, do not call `mujoco.viewer` or `mujoco.Renderer` in-process; MuJoCo needs an
OpenGL backend (`MUJOCO_GL=egl|osmesa|glfw`) that this project does not require. `parkour watch`
reports this cleanly, and the browser replay path is the supported way to look at motion.

## Control loop

| Setting | Default | Meaning |
|---|---|---|
| `physics_timestep` | 0.004 s | MuJoCo integration step |
| `control_decimation` | 5 | physics steps per control step |
| `control_dt` | 0.02 s (20 Hz) | derived: `physics_timestep × control_decimation` |
| `action_scale` | 0.25 | half-range of each joint target, in units of the joint's radian range |
| `action_smoothing` | 0.6 | first-order filter on actuator targets (0 = raw) |
| `max_episode_seconds` | level limit (20–50 s) | optional shorter cap, applied to training only |

An action of `0` holds the standing pose. A value of `1` asks for
`(upper - lower) / 2 × action_scale` radians away from the standing pose on that joint, always
clamped to the joint's mechanical range. The mapping is centred, so the standing pose is exactly
reachable and no action can drive a joint past its limit.

## Observation (90 values)

| Block | Count | Contents |
|---|---|---|
| root | 12 | support-relative pelvis height, up-vector z, body-frame up vector (3), world linear velocity (3), body angular velocity (3), absolute pelvis height |
| goal | 4 | goal − pelvis position (3, scaled), remaining distance |
| task | 5 | progress fraction, time remaining, floor height ahead, next obstacle rise, distance to the next void |
| obstacles | 12 | three upcoming solids: dx, dy, rise above support, half-width |
| proprio | 57 | joint positions (19, normalised to limits), joint velocities (19), previous action (19) |

Observations are normalised online (`RunningNorm`) inside the policy; raw values are clipped by the
environment so a diverged simulation cannot produce non-finite network inputs.

## Reward

Every component is reported per step in `info["reward_components"]`, and the sum equals the step
reward. Weights live in `RewardConfig` (`src/parkour_lab/env.py`) and can be overridden from the CLI
or the dashboard's API without editing code.

| Key | Default | Notes |
|---|---|---|
| `progress_per_metre` | 5.0 | potential-based: only the change in distance to the finish pays |
| `forward_velocity_per_second` | 0.0 | `min(vx / target, 1)`, target `forward_velocity_target=1.2` |
| `forward_velocity_target` | 1.2 m/s | saturating speed used by the velocity and gait terms |
| `gait_per_second` | 0.0 | credits a clean left/right foot exchange, gated on forward speed |
| `alive_per_second` | 1.0 | survival |
| `upright_per_second` | 0.5 | torso orientation (zero below a 0.55 up-vector) |
| `height_per_second` | 1.0 | closeness to the standing pelvis height above the support |
| `action_rate_penalty` | 0.01 | mean squared change between consecutive actions |
| `torque_penalty` | 1e-6 | mean squared actuator force |
| `joint_velocity_penalty` | 1e-5 | mean squared joint velocity |
| `lateral_drift_penalty` | 0.02 | sideways speed |
| `success_bonus` | 40 | terminal, on `course_complete` |
| `speed_bonus` | 6 | terminal, scaled by remaining time |
| `fall_penalty` / `hazard_penalty` / `out_of_bounds_penalty` | 10 / 12 / 8 | terminal |

The total is multiplied by `--reward-scale` (default 0.25) before it reaches PPO, which keeps the
critic in a well-conditioned range.

### Reward annealing

A dense forward-progress reward from step 0 has a strong local optimum: lunge at the finish line
and fall, because falling forward earns progress before the episode ends. Two mechanisms exist to
avoid it:

```bash
--reward-ramp-fraction 0.1          # first 10% of the run anneals from the start reward
--reward-start upright_per_second=4 # override one starting weight (posture-only preset)
```

The start preset is `RewardConfig.posture_stage()`: progress, velocity and gait are 0, while
alive/upright/height are emphasised. Both weight sets and the ramp fraction are written to
`config.json`, so the run is reproducible and `TrainConfig.from_dict` restores it.

## Training

```bash
parkour train --level level-01-basics --all --curriculum adaptive ...
```

* `--level` is repeatable. `--all` selects the 20 training levels (the two held-out levels are
  excluded unless `--held-out` is passed explicitly).
* `--curriculum single|uniform|adaptive`. `adaptive` tracks per-level success/completion in an
  EWMA, samples weak levels more often, keeps mastered levels in the mix for periodic retesting
  and gives unvisited levels a novelty bonus (`curriculum.CurriculumConfig`).
* `--envs`, `--workers`: number of environments and subprocess workers. Defaults follow the measured
  benchmark on a 2-vCPU machine (`num_envs=4`, `workers=1`).
* `--steps`, `--rollout-steps`, `--epochs`, `--lr`, `--entropy`, `--gamma`, `--hidden-dim`.
* `--eval-interval`, `--eval-levels`, `--eval-seeds`, `--eval-episodes`,
  `--checkpoint-interval`, `--keep-checkpoints`, `--max-replays`.
* `--max-minutes` is a hard wall-clock budget; the trainer checkpoints and exits cleanly.
* `--resume runs/<id>/checkpoints/latest.npz` continues a run (policy, normaliser, optimiser-free
  state, iteration counter and sampler state).

Training writes everything described in [ARCHITECTURE.md](ARCHITECTURE.md#run-directory-layout-implemented).
`runs/` is ignored by Git.

## Evaluation protocol

`parkour evaluate` runs the policy on the given levels and seeds with:

* deterministic actions (`mean` of the Gaussian) unless `--stochastic` is passed;
* the dynamics recorded in the checkpoint — action scale, smoothing, timestep, decimation,
  actuation mode and reward;
* reset noise per seed (`EvalConfig.randomize_reset=True`, the default), so several seeds measure
  robustness to the initial state instead of replaying the same episode N times.

Reported per level: episodes, success rate, mean/σ completion, mean return, mean duration, and the
termination histogram. `parkour compare --run-a A --run-b B` prints the paired per-level difference
between the best evaluation of each run.

## Replays

`--record-replay` (evaluate) or `parkour replay` writes gzipped JSON with one entry per captured
frame:

```json
{"level": {...}, "level_id": "level-01-basics", "seed": 11,
 "frames": [{"t": 0.0, "bodies": {"pelvis": [x, y, z, qw, qx, qy, qz], "...": []}, "movers": {}}]}
```

Frames are pure data: no rendering, no display, no GPU. The dashboard plays them in the browser with
Three.js, which is why a headlessly trained agent can be inspected on the same CPU-only host that
trained it. `frame_stride` controls capture density.

## Reproducibility checklist

* `config.json` + `provenance.json` (git commit, dirty flag, versions, CPU/GPU) are written at start.
* Seeds: `TrainConfig.seed` seeds the policy init and environment seeds (`seed + 1000·i`).
* Evaluation and replay are deterministic given the checkpoint and seed.
* `tests/test_training_pipeline.py::test_training_is_reproducible` trains twice and compares the
  metrics byte-for-byte for the same seed.
