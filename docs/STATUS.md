# Status report — 2026-10-24

This is the honest state of the project: what works, what was measured, what does not work yet,
and how to reproduce every claim. It is the document to read before trusting any other summary.

## 1. What runs today

| Capability | Command | Verified by |
|---|---|---|
| Environment check (versions, CPU, MuJoCo, 22 levels) | `parkour doctor` | `tests/test_cli.py::test_doctor_reports_a_usable_environment` |
| Level catalogue | `parkour list-levels [--json]` | `tests/test_cli.py::test_list_levels_text_and_json` |
| Level/humanoid inspection, MJCF export | `parkour inspect`, `parkour export-mjcf` | `tests/test_cli.py` |
| Scripted/random rollouts | `parkour random` | `tests/test_cli.py::test_random_policy_runs_and_reports` |
| PPO training with checkpoints, resume, replays | `parkour train` | `tests/test_training_pipeline.py` (real 1 024-step run, resume, reproducibility) |
| Deterministic multi-seed evaluation | `parkour evaluate` | `tests/test_cli.py`, `tests/test_training_pipeline.py` |
| Replay export (data only, no rendering) | `parkour replay` | `tests/test_cli.py::test_replay_command_writes_a_frame_log` |
| Run listing and paired comparison | `parkour runs`, `parkour compare` | `tests/test_cli.py` |
| Throughput benchmark | `parkour benchmark` | `tests/test_cli.py::test_benchmark_runs_a_short_sweep` |
| Dashboard (API + 3D replay viewer) | `parkour dashboard` | `tests/test_dashboard.py` (14 tests, incl. real job start/stop) |
| Bounded CEM over reward/PPO knobs | `parkour evolve` | `tests/test_research_tools.py`, `tests/test_cli.py` |
| Restricted research assistant | `parkour harness status/plan/probe` | `tests/test_research_tools.py` |

Test suite: **143 tests** (`pytest`, ~42 s) and `ruff check src tests` clean.

## 2. The model and the task

* Humanoid: 25 DoF, 19 actuated (both legs 5+5, spine 2, neck 1, arms 3+3), 46.6 kg, 26 qpos.
  Position servos with explicit `force_limit` per joint; an activation-filtered torque mode is
  available as an experiment, not as the default.
* Control: 19 normalised actions in `[-1, 1]`, centred on the standing pose, scaled by
  `action_scale` (default 0.25) of each joint's radian half-range, optionally smoothed by a
  first-order filter (`action_smoothing=0.6`).
* Observation: 90 values (root 12, goal 4, task 5, obstacles 12, proprio 57), documented by name
  in `OBSERVATION_NAMES`.
* Terminations: `course_complete`, `fell_over`, `collapsed`, `void_fall`, `hazard_contact`,
  `out_of_bounds`, `timeout`, and optionally `no_progress`.
* Levels: 20 training + 2 held-out, validated against `configs/levels/level.schema.json`, generated
  deterministically by `tools/generate_levels.py` (`--check` reproduces them byte-identically).

## 3. Measured throughput (2 vCPU, no GPU, no CUDA)

Measured with `parkour benchmark` (control steps per second, 4 sub-processes are slower than serial
on this host because of inter-process overhead):

| Configuration | Throughput |
|---|---|
| 1 environment | ~3.2–3.4 k steps/s |
| 4 serial environments | ~1.7–2.2 k steps/s per-environment-clock, ~4 k aggregate in the benchmark |
| 4 environments, threads | ~0.5× of serial (GIL) |
| 2 worker processes, 4 envs | ~3.4 k steps/s |

Defaults chosen from these numbers: `num_envs=4`, `workers=1`, `physics_timestep=0.004`,
`control_decimation=5` (50 Hz control, the value that trains measurably faster than 100 Hz).

MJWarp/mjlab were **not** installed or benchmarked — this host has no GPU/CUDA, so no claim is made
about them.

## 4. Learning status

Primary run: `runs/walk-e` (retained locally; `runs/` is git-ignored — regenerate with the command
in §6).

Configuration: `level-01-basics` (6 m runway, one 10 cm step), 4 envs, 50 Hz control,
`action_scale=0.3`, `action_smoothing=0.6`, reward annealed over the first 10 % of the run from
posture-only shaping (`posture_stage()`) to progress 6/m + forward velocity 4/s (target 1 m/s) +
gait 6/s + upright/height/alive shaping, `no_progress` termination after 2 s, episode cap 10 s.

Results at 1.7 M steps (evaluation is now closed-loop — see the bug table in §5):

| Metric | Value |
|---|---|
| Success rate | **0 %** |
| Deterministic evaluation, 5 seeds (11/23/37/41/53) | **45.5 %** mean completion ≈ 2.7 m |
| Stochastic evaluation, same 5 seeds | 45.6 % — the mean action is not "more conservative" |
| Run's own 2-seed rolling eval at 1.72 M steps | 58 % completion, mean return 54 |
| Training episodes, last 300 | 49 % mean completion ≈ 2.9 m in 3.8 s |
| Termination mix (last 300) | 95 % `collapsed`, 5 % `fell_over` |
| Transfer to the next level (`level-02-step-up`, 3 seeds) | 12 % completion — no completed level |
| Explained variance / approx. KL | ~0.95 / ~0.02 (the learner is healthy; the task is not solved) |

Interpretation: PPO reliably learns to stand, to survive longer, and to convert that into forward
motion — 0 → 2.7 m over ~1.7 M steps — but every episode still ends in `collapsed` before the 6 m
finish. There is **no** walking gait and **no** completed level, so there is nothing to report on
levels 3–22. The two held-out levels have never been used for tuning.

> **Numbers from earlier runs are not comparable.** Until the evaluation loop was fixed (§5, last
> row), `parkour evaluate` reported an open-loop controller and understated every score by roughly
> 4×: the same `walk-e` checkpoint scored 10.3 % before the fix and 45.5 % after it. Evaluation rows
> in `runs/walk-e/metrics/evaluation.csv` below ~1.44 M steps, the CEM fitness values recorded in
> `runs/*/harness2/`, and the "13–21 % completion" figures quoted in the earlier revision of this
> document were all measured with the broken loop and should be ignored. Training metrics
> (`metrics/training.csv`, `episodes.jsonl`) were never affected — the trainer always acted on fresh
> observations.

### Dead ends that are documented rather than hidden

1. **Dense progress reward from step 0** → "lunge at the finish and fall": return rises while
   completion plateaus, every episode ends in `collapsed`/`fell_over` after ~2.5 s. Mitigated with
   reward annealing from a posture-only stage (`--reward-ramp-fraction`).
2. **Gait shaping without no-progress termination** → "march in place": 82 of 120 episodes timed out
   with 2.4 % completion while collecting gait credit. Fixed with `--stagnation-seconds` and by
   gating gait credit on actual forward speed.
3. **Scripted CPG search** (≈ 2 000 open-loop parameter combinations over hip/knee/ankle amplitude,
   frequency, phase and lean): best forward displacement 0.36 m before falling. This suggests the
   remaining gap is not only sample budget; morphology/actuation (foot size, ankle authority,
   straight-knee default pose) is a plausible contributor.
4. **Bigger feet / wider stance / coarser timestep** were tested as fixes for stability and were not
   the bottleneck at the time — but those measurements were taken while the action-span bug below
   was still active, so they need to be re-run with the current model.

## 5. Bugs found and fixed (with regression tests)

| Bug | Effect | Test |
|---|---|---|
| Action span computed from degree limits and written into radian `ctrl` | ~57× action amplification; every early stability result was an artifact | `test_humanoid_levels.py` action-span test |
| Observation block overwritten (`obs[0:3]` then `obs[2:5]`) | root position silently replaced by the body up-vector | `test_observation_blocks_do_not_overwrite_each_other` |
| `rollout_episode` rebuilt `EnvConfig` by hand when recording frames | evaluation/replay silently dropped action scale, smoothing, actuation and reward | `test_evaluation_rebuilds_checkpoint_dynamics` |
| Evaluation used a deterministic policy with a fixed reset | all seeds produced the identical episode; multi-seed numbers were one sample | `test_eval_seeds_probe_different_resets` |
| `EnvConfig.physics_timestep` never applied to `mjModel.opt.timestep` | "50 Hz" runs were actually 100 Hz; time-based reward, episode caps and stagnation timers were all off by 2× | `test_config_timestep_is_authoritative` |
| `list_runs` had no flat summary keys | CLI/dashboard showed empty steps and success | `test_run_list_and_detail_contract` |
| Checkpoint `extra` lacked `run_id`/`level_ids` | resume and provenance could not identify their run | `test_training_writes_every_expected_artifact` |
| Killed trainers kept `status: running` | stale runs looked live in the CLI and dashboard | `test_stale_running_run_is_reported_as_interrupted` |
| `evolution.default_vector` iterated dict keys | `NameError` on the first CEM search | `test_default_vector_reproduces_the_defaults` |
| Run list showed `summary.total_steps`, i.e. the *budget* recorded at startup | a live 2.5 M-step run displayed as "5,000,000 steps" and a finished-looking card; killed runs without a pid also showed "running" forever | `test_run_list_separates_progress_from_the_step_budget`, `test_running_run_without_pid_goes_stale` |
| `rollout_episode` never wrote `result.observation` back into the loop variable | every `parkour evaluate` episode, replay and Harness²/CEM score ran the policy open-loop on the reset frame — 10.3 % instead of 45.5 % completion for the same checkpoint | `test_rollout_episode_is_closed_loop`, `test_rollout_episode_matches_a_manual_loop` |

## 6. How to reproduce

```bash
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
pytest && ruff check src tests

# the run behind the numbers in §4
parkour train --level level-01-basics --run-id walk-e --steps 5000000 --envs 4 \
  --rollout-steps 512 --seed 5 --physics-timestep 0.004 --decimation 5 \
  --action-scale 0.3 --action-smoothing 0.6 --init-log-std -1.0 --entropy 0.0015 \
  --lr 0.0003 --epochs 6 --gamma 0.99 --max-episode-seconds 10 --stagnation-seconds 2 \
  --reward-ramp-fraction 0.1 --reward-set progress_per_metre=6 \
  --reward-set forward_velocity_per_second=4 --reward-set gait_per_second=6 \
  --reward-set alive_per_second=1 --reward-set upright_per_second=1.5 \
  --reward-set height_per_second=2 --reward-set action_rate_penalty=0.02 \
  --reward-set fall_penalty=2 --eval-interval 20 --eval-episodes 2

# inspect it
parkour evaluate --checkpoint runs/walk-e/checkpoints/best.npz --level level-01-basics \
  --eval-seeds 11 23 37 41 53
parkour dashboard --port 8000     # http://localhost:8000 → Replay tab
```

## 7. What remains unfinished

1. **Learning to walk / complete levels.** Highest priority, and the first job now that evaluation
   is trustworthy is to re-baseline: the same `walk-e` checkpoint was reported at 10.3 % throughout this
   build and is really 45.5 %, so morphology/actuation conclusions drawn from the old
   numbers are unproven. Concrete next experiments, in the order I would run them:
   * Re-run the morphology/actuation sweep now that the action-span and timestep bugs are fixed:
     larger feet, bent-knee default stance, stronger ankle torque, 50 Hz vs 100 Hz control.
   * Train from a **reference gait** (a scripted gait retargeted as a tracking objective) instead of
     pure exploration — the CPG experiments suggest exploration alone is not finding a stable
     contact cycle.
   * Add **mirror symmetry** to the observation/action space and symmetric data augmentation, which
     typically halves the effective sample requirement for locomotion.
   * Increase the sample budget by an order of magnitude (10–50 M steps) on better hardware, or port
     the same MJCF to a batched GPU backend and benchmark it (MJWarp/mjlab remain unverified here).
2. **Multi-seed baselines.** Only one long training seed has been run end-to-end; `evaluate`,
   `compare` and the evolve tool exist and are tested, but the science needs 3+ seeds.
3. **Per-level training evidence.** The adaptive curriculum is implemented and exercised on a 3-level
   smoke run, but no multi-level run has been trained long enough to say anything about levels 2–22.
4. **Experience / hybrid learning.** No experience retrieval layer; the CEM tuner exists and has
   been exercised on a short real sweep (6 individuals × 3 generations × 40 k steps), but that
   sweep's fitness came from the open-loop evaluation described in §5, so it needs re-running
   before its result means anything.
5. **Harness² network planner.** Works only if a local endpoint is configured; disabled by default
   and intentionally untested against a live model.
6. **Native viewer.** `parkour watch` requires an OpenGL display; on this host it reports the
   limitation instead of crashing. Browser replay is the supported visual path.
