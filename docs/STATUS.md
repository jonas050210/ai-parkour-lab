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
| Failure analysis of a checkpoint (hop/stand/collapse, per seed) | `parkour diagnose` | `tests/test_diagnostics.py` (5 tests), `tests/test_cli.py::test_diagnose_command_explains_a_checkpoint` |
| Replay export (data only, no rendering) | `parkour replay` | `tests/test_cli.py::test_replay_command_writes_a_frame_log` |
| Run listing and paired comparison | `parkour runs`, `parkour compare` | `tests/test_cli.py` |
| Throughput benchmark | `parkour benchmark` | `tests/test_cli.py::test_benchmark_runs_a_short_sweep` |
| Dashboard (API + 3D replay viewer) | `parkour dashboard` | `tests/test_dashboard.py` (14 tests, incl. real job start/stop) |
| Bounded CEM over reward/PPO knobs | `parkour evolve` | `tests/test_research_tools.py`, `tests/test_cli.py` |
| Restricted research assistant | `parkour harness status/plan/probe` | `tests/test_research_tools.py` |

Test suite: **172 tests** (`pytest`, ~36 s) and `ruff check src tests` clean.

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

Results for the completed 5 M-step run (`walk-e`: 2 442 iterations, 27 328 episodes; 3.56 M of
those steps took 33 min after resuming at 1.44 M, i.e. ~1 780 steps/s on 2 vCPU):

| Metric | Value |
|---|---|
| Success rate | **0 %** (0 of 27 328 training episodes, 0 of every evaluation) |
| Deterministic evaluation, 5 seeds (11/23/37/41/53) | **47.7 %** mean completion ≈ 2.9 m |
| Stochastic evaluation, same 5 seeds | 46.9 % — sampling noise is not the limiting factor |
| Run's own 2-seed rolling evaluation, final | 58.4 % completion, mean return 51.3 |
| Training episodes, last 300 | 51 % mean completion ≈ 3.1 m in 3.55 s |
| Termination mix (last 300) | 92 % `collapsed`, 8 % `fell_over` |
| All 20 training levels, 2 seeds each | 13.8 % mean completion, 0 % success; per level 6.3–14.8 % |
| Explained variance / approx. KL | 0.98 / 0.016 (the learner is healthy; the task is not solved) |

The plateau is a *reward-geometry* result, confirmed by a controlled counter-experiment. `walk-f`
is the same recipe with survival priced up (fall penalty −20 instead of −2, alive/upright/height
weights doubled) and 1.5 M steps:

| Run | Fall penalty | Survival weights | Episode length | Completion (5 seeds) |
|---|---|---|---|---|
| `walk-e` | −2 | 1.0 / 1.5 / 2.0 | 3.55 s | 47.7 % |
| `walk-f` | −20 | 2.0 / 2.5 / 2.5 | 7.64 s | 21.3 % |

Both end at 0 % success. Pricing survival higher buys exactly what it says — the policy stands and
shuffles (48 % of `walk-f` episodes now reach the 10 s cap, terminations flip from `collapsed` to
`timeout`) — and buys it with progress. Neither setting makes the agent *walk*, so the missing
ingredient is not a weight on the existing reward terms.

`walk-g` attacks the observed *hopping* directly: it adds an opt-in flight-phase penalty (charged
while both feet are in the air) and a `gait_speed_floor` so a policy that cannot move yet still
gets a gradient for stepping. It was stopped by the operator at 3.52 M of its 6 M budget, and what
it shows is a faster route to the *same* ceiling, not a higher one:

| Steps | `walk-e` rolling eval | `walk-g` rolling eval |
|---|---|---|
| 2.01 M | 58.8 % | 53.9 % |
| 3.48 M | 59.2 % | 47.6 % |

At 3.52 M steps `walk-g` evaluates at 42.1 % over 5 seeds (training-episode mean 45.7 %, 4.63 s,
85 % `collapsed`) — inside the same band as `walk-e` at the same budget. So the shaping terms are
worth keeping as tools (they are off by default and unit-tested), but they do not change the
qualitative result.

**The policy plateaued, so this is not a sample-budget story alone.** Completion sat at 58.8 %
at 1.68 M steps, 59.4 % at 3.40 M and 58.4 % at 5.0 M: 3.3 M further steps (~30 min of CPU) bought
nothing. The learned behaviour is a fast, symmetric hop that covers ~3 m in 4.5 s at ~0.8 m/s
(21 % of control steps have both feet off the ground) and then collapses. The policy is healthy
(value loss falls, explained variance 0.98, KL 0.016) and the *task* reward is what saturates: with
`progress 6/m` and `fall_penalty 2`, falling forward after 3 m already pays better than walking
slowly, and the reward never distinguishes a hopping controller from a walking one.

Where the 5 M-step policy actually breaks down is visible in one rollout: it accelerates to
1.3 m/s, cruises at 0.8–1.0 m/s, and collapses 0.16 m past the far edge of the 10 cm step
(`x = 3.56 m`, pelvis height dropping 0.95 → 0.80 m over the last 15 control steps). The
observation is terrain-conditioned (obstacle-ahead features at indices 25–28), so this is a
controller tuned to `level-01`'s geometry, not a general gait.

There is **no** walking gait and **no** completed level, so there is nothing to report on levels 3–22
beyond "the level-01 policy transfers weakly and falls over": the 20-level sweep ends in `fell_over`
26/40 and `collapsed` 14/40. The two held-out levels have never been used for tuning.

> **Numbers from earlier runs are not comparable.** Until the evaluation loop was fixed (§5),
> `parkour evaluate` reported an open-loop controller and understated every score by roughly 4×:
> the same `walk-e` checkpoint scored 10.3 % before the fix and 45.5 % after it. Evaluation rows in
> `runs/walk-e/metrics/evaluation.csv` below ~1.44 M steps, the CEM fitness values recorded in
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
5. **Survival-first reward** (`walk-f`: fall penalty −20, survival weights doubled): the agent stopped
   falling by mostly standing. 7.64 s episodes instead of 3.55 s, 21.3 % completion instead of
   47.7 %, still 0 % success. Moving the survival/progress ratio only moves the plateau.

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
| "Best evaluation" ranked on success only | every run with 0 % success reported its *first* evaluation as best (13 % instead of 59 % for `walk-e`) | `test_best_evaluation_breaks_success_ties_on_completion` |
| `rollout_episode` never wrote `result.observation` back into the loop variable | every `parkour evaluate` episode, replay and Harness²/CEM score ran the policy open-loop on the reset frame — 10.3 % instead of 45.5 % completion for the same checkpoint | `test_rollout_episode_is_closed_loop`, `test_rollout_episode_matches_a_manual_loop` |
| Rollout actions were sampled outside the control range and scored there | the env clips to [-1, 1]; 24.6 % of action dimensions in a trained rollout (99 % of steps) were clipped, so PPO stored log-probabilities for actions that were never applied | `test_sampled_actions_stay_inside_the_control_range` |
| `benchmark_simulation` ignored `workers` and the CLI overwrote the label | `parkour benchmark --workers` printed configurations it never ran, and the "best measured configuration" could recommend an unmeasured layout | `test_benchmark_simulation_measures_the_requested_worker_layout` |
| `--eval-episodes N` was truncated to `len(seeds)` | asking for 8 episodes with 3 seeds silently ran 3 | `test_evaluation_runs_the_requested_number_of_episodes` |
| Finished episodes were labelled with the level assigned for the *next* reset | per-level statistics (and therefore the adaptive curriculum) could be attributed to the wrong level | `test_finished_episode_is_labelled_with_the_level_it_ran` |
| A successful episode reported ~85 % completion | `completion` is a raw distance ratio while success only needs the finish disc | `test_successful_episode_reports_full_completion` |
| `env_config_from_checkpoint` hard-coded `randomize_reset=False` | any tool that did not override it replayed three identical "seeds" | `test_diagnose_uses_every_seed` |
| Checkpoint `episode_stats` was declared, never written, and serialised with `float(value)` | the key either held an empty list or would have raised `TypeError`; a checkpoint could not describe the episodes around it | `test_checkpoint_carries_recent_episode_outcomes` |
| `parkour watch --checkpoint` rebuilt the default `EnvConfig` | the GUI played a checkpoint with the wrong action scale, smoothing, timestep or actuation while `evaluate`/`replay` used the recorded ones | `test_watch_uses_the_checkpoint_dynamics` |
| `parkour watch` on a headless machine called `exit(1)` inside MuJoCo | importing `mujoco.viewer` initialises GLFW, which terminates the process from C — the friendly error message was unreachable | `test_watch_refuses_a_headless_machine_without_killing_the_process` |
| The trainer acted once per episode on the *previous* episode's final observation | `step()` returns the state that ended the episode and the reset happens afterwards, so the first action of every episode was paired with the wrong state (one corrupted transition per episode, and a policy decision made on a finished episode) | `test_vector_autoreset_returns_the_post_reset_observation`, `test_rollout_acts_on_the_reset_observation_after_an_episode_ends` |
| Truncated episodes were bootstrapped as if they were terminal | a time limit or stagnation stop cut the value target to `r`, teaching the critic that long episodes are worthless; `V(s_final)` is now used | `test_gae_bootstraps_a_truncated_episode` |
| `--target-success 0` (and any target) stopped the run after the *first evaluation* | "stop at the first success" could not be expressed: the first evaluation sets `best_eval_success` to 0.0 even with zero successes, and `0.0 >= 0.0` ended the run | `test_stop_reason_needs_a_real_success`, `test_stop_on_success_ends_the_run_at_the_first_completed_course` |
| A run's summary could not say why it ended, and `best_eval_success` was -1.0 before the first evaluation | a bounded or early-stopped run was indistinguishable from a crash in `summary.json` | `test_runs_without_stop_on_success_spend_their_budget` |

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

1. **Learning to walk / complete levels.** Highest priority. The 5 M-step run rules out two easy
   explanations: it is not evaluation noise (deterministic ≈ stochastic) and it is not sample
   budget alone (3.3 M extra steps changed nothing). The remaining suspects are reward geometry and
   exploration structure. Concrete next experiments, in the order I would run them:
   * **Reward geometry is exhausted as a lever.** `walk-e` (fall −2) hops 3 m and collapses;
     `walk-f` (fall −20, survival doubled) stands and shuffles for 7.6 s instead; `walk-g`
     (flight-phase penalty + gait floor) reaches the same 55–59 % ceiling in half the steps. All
     three score 0 %. Re-tuning these weights again is not the next move — the missing signal is
     *how* to step, which points at a reference motion rather than at another scalar weight.
   * **Reference gait tracking** instead of pure exploration: the CPG search over ~2 000 open-loop
     parameter sets peaked at 0.36 m, so the reference should come from a scripted quasi-static
     walk validated in the *enclosed* loop, or from a public retargeted mocap clip.
   * **Mirror symmetry** in the observation/action space plus symmetric augmentation, which
     typically halves the samples needed for locomotion.
   * **Re-run the morphology/actuation sweep** (larger feet, bent-knee stance, ankle authority,
     50 Hz vs 100 Hz) now that the action-span, timestep and evaluation bugs are all fixed — the
     earlier sweeps were taken under at least one of those bugs.
   * **Bigger budgets on better hardware** (10–50 M steps, or a batched GPU backend benchmarked
     against this one; MJWarp/mjlab remain unverified here).
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
