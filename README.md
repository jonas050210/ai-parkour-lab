# AI Parkour Lab

A local-first research project for teaching a physically simulated humanoid to traverse progressively harder parkour courses. The long-term system combines articulated-body physics, reinforcement learning, evolutionary search, reusable experience, persistent experiment records, and a 3D research dashboard.

> **Status — research and foundation phase (2026-10-23).** The repository currently contains the research plan, initial interfaces, configuration/level templates, and documentation. It does **not** yet contain a physics environment, a trainable policy, a dashboard, a trained agent, or a working Harness² integration. No training has been run.

## Research goals

- Start with a randomly initialized policy and measure how it learns, rather than silently giving it a parkour policy or motion-tracking controller.
- Simulate a complete articulated humanoid and make actuation limits, contact, falls, and obstacle interactions consequential.
- Progress through fixed, named levels: safe movement and low steps, jumps and gaps, hazards, moving surfaces, advanced transitions, and height exposure.
- Compare algorithms and curriculum changes with seeded, held-out evaluation—not only a single best run.
- Keep policies, checkpoints, course definitions, metrics, and experiment provenance on disk.
- Show the course and selected runs in a responsive 3D viewer, while keeping the dashboard out of the training hot path.

## Proposed direction

| Concern | Starting recommendation | Status |
|---|---|---|
| Physics/model authoring | MuJoCo MJCF; validate one environment with ordinary MuJoCo first | Research recommendation; not implemented |
| Batched NVIDIA training | Evaluate MuJoCo Warp through `mjlab`; start small and profile the actual humanoid | Candidate backend; not installed or hardware-verified |
| Initial learning | PPO with joint-target/torque-limited control; evolve outer-loop parameters after a reproducible PPO baseline | Research recommendation; not implemented |
| Experiment records | Local run directories, versioned configs, JSONL summaries, checkpoints, and TensorBoard | Planned |
| 3D dashboard | React + TypeScript + Three.js / React Three Fiber, served alongside a local API | Planned |
| Harness² | Optional developer/research-agent experiment harness, separate from the simulated controller and RL optimizer | Investigated; not integrated |

The intended development machine is Linux with an NVIDIA RTX 4060 Ti (8 GB VRAM) and 32 GB RAM. The 8 GB limit is material: the plan begins with headless training and a small measured environment batch, leaves memory headroom, and only scales after profiling. The actual CUDA driver, GPU memory availability, and supported package combination must be checked on the target computer; results are not guaranteed by this plan.

## Repository map

```text
configs/
  default.toml                 Future run defaults (not yet consumed by an application)
  levels/
    level.schema.json          Versioned JSON contract for fixed courses
    level-01-basics.json       Example easy course manifest, not a simulation
src/parkour_lab/
  contracts.py                 Dependency-free boundaries for later components
  __init__.py
docs/
  RESEARCH.md                  Source-checked comparisons and conclusions
  ARCHITECTURE.md              Components, data flow, and Harness² boundary
  ROADMAP.md                   Phases and acceptance criteria
  DECISIONS.md                 Decision log and open questions
tests/
  test_foundation.py           Checks the starter config and example manifest
```

Generated runs, datasets, TensorBoard event files, and model weights belong in ignored `data/` or `runs/`, not Git. See [docs/RESEARCH.md](docs/RESEARCH.md) for evidence and [docs/ROADMAP.md](docs/ROADMAP.md) for the next steps.

## Foundation checks

Python 3.12 is the planned development baseline. The metadata, interfaces, and templates use only the standard library; the foundation tests use the optional test extra (pytest, Ruff, and JSON Schema validation):

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The optional `sim`, `training`, and `web` extras describe later integration points; they are not required for these checks. Installing an extra alone does not create an environment, trainer, or web application. See `pyproject.toml` and [docs/DECISIONS.md](docs/DECISIONS.md) before choosing a backend or adding a large dependency stack.

## Honest scope

This repository is a plan and a small software foundation, not a playable game or an implemented AI lab. Success criteria, model fidelity, training speed, algorithm comparisons, and local GPU compatibility remain to be established by the implementation phases. The level manifest is a data contract only; it does not spawn geometry or define physics by itself.
