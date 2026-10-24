#!/usr/bin/env python3
"""Generate the fixed parkour level manifests in ``configs/levels``.

The generator is the authoring tool for the 20 training levels plus two held-out
generalisation levels. It is deterministic: running it reproduces byte-identical JSON.
Geometry uses MuJoCo conventions: ``position`` is the box centre, ``size`` the half
extents, +x forward, +y left, +z up, and a "gap" is simply absent geometry annotated with
a non-solid ``gap`` obstacle.

Run::

    python tools/generate_levels.py            # write manifests
    python tools/generate_levels.py --check    # verify committed files are current
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS_DIR = ROOT / "configs" / "levels"
PELVIS_HEIGHT = 0.92
SPAWN_CLEARANCE = 0.02


def _round(value: float) -> float:
    return round(value + 0.0, 4)


def slab(
    oid: str,
    x0: float,
    x1: float,
    top: float,
    width: float = 2.4,
    *,
    y: float = 0.0,
    bottom: float = -0.4,
    kind: str = "platform",
    hazard: bool = False,
    motion: dict | None = None,
    rotation: float | None = None,
) -> dict:
    """Axis-aligned box from x0..x1 with its top surface at ``top``."""

    assert x1 > x0, oid
    assert width > 0, oid
    assert top > bottom, oid
    entry: dict = {
        "id": oid,
        "kind": kind,
        "position": [_round((x0 + x1) / 2), _round(y), _round((top + bottom) / 2)],
        "size": [_round((x1 - x0) / 2), _round(width / 2), _round((top - bottom) / 2)],
    }
    if hazard:
        entry["hazard"] = True
    if motion is not None:
        entry["motion"] = motion
    if rotation:
        entry["rotation_degrees"] = rotation
    return entry


def box(
    oid: str,
    x_center: float,
    top: float,
    *,
    length: float,
    width: float,
    y: float = 0.0,
    bottom: float = -0.4,
    kind: str = "platform",
    hazard: bool = False,
    motion: dict | None = None,
) -> dict:
    return slab(
        oid,
        x_center - length / 2,
        x_center + length / 2,
        top,
        width,
        y=y,
        bottom=bottom,
        kind=kind,
        hazard=hazard,
        motion=motion,
    )


def gap(oid: str, x0: float, x1: float, width: float = 2.4, y: float = 0.0) -> dict:
    """Annotation only: describes a hole in the course for tooling and the dashboard."""

    return {
        "id": oid,
        "kind": "gap",
        "position": [_round((x0 + x1) / 2), _round(y), 0.0],
        "size": [_round((x1 - x0) / 2), _round(width / 2), 0.02],
    }


def stairs(
    prefix: str,
    x0: float,
    *,
    count: int,
    rise: float,
    run: float,
    start_top: float,
    width: float,
    ascending: bool = True,
) -> list[dict]:
    steps = []
    for index in range(count):
        step_top = start_top + rise * (index + 1 if ascending else -index)
        steps.append(
            slab(
                f"{prefix}-{index + 1}",
                x0 + run * index,
                x0 + run * (index + 1),
                step_top,
                width,
                kind="step",
            )
        )
    return steps


def make_level(
    *,
    level_id: str,
    title: str,
    order: int,
    difficulty: int,
    summary: str,
    spawn_x: float,
    spawn_top: float,
    finish_x: float,
    finish_radius: float,
    obstacles: list[dict],
    time_limit: float,
    par_time: float,
    tags: list[str],
    split: str = "train",
    bounds_y: float = 4.0,
    bounds_z: tuple[float, float] = (-3.0, 5.0),
) -> dict:
    finish_top = _support_top(obstacles, finish_x, 0.0)
    xs = [spawn_x, finish_x]
    for obstacle in obstacles:
        xs.extend(
            [
                obstacle["position"][0] - obstacle["size"][0],
                obstacle["position"][0] + obstacle["size"][0],
            ]
        )
    max_x = max(xs) + 2.5
    min_x = min(xs) - 2.5
    return {
        "schema_version": 1,
        "id": level_id,
        "title": title,
        "order": order,
        "difficulty": difficulty,
        "split": split,
        "tags": tags,
        "summary": summary,
        "time_limit_seconds": time_limit,
        "par_time_seconds": par_time,
        "units": {"distance": "m", "angle": "degree", "up_axis": "+z", "forward_axis": "+x"},
        "spawn": {
            "position": [_round(spawn_x), 0.0, _round(spawn_top + PELVIS_HEIGHT + SPAWN_CLEARANCE)],
            "yaw_degrees": 0.0,
        },
        "finish": {
            "position": [_round(finish_x), 0.0, _round(finish_top + PELVIS_HEIGHT)],
            "radius": finish_radius,
        },
        "bounds": {
            "minimum": [_round(min_x), -bounds_y, bounds_z[0]],
            "maximum": [_round(max_x), bounds_y, bounds_z[1]],
        },
        "obstacles": obstacles,
    }


def _support_top(obstacles: list[dict], x: float, y: float) -> float:
    best = -math.inf
    for obstacle in obstacles:
        if obstacle["kind"] == "gap":
            continue
        px, py, pz = obstacle["position"]
        sx, sy, sz = obstacle["size"]
        if abs(x - px) <= sx and abs(y - py) <= sy:
            best = max(best, pz + sz)
    if not math.isfinite(best):
        raise SystemExit(f"no support at x={x}, y={y}")
    return best


def level_01() -> dict:
    obstacles = [
        slab("runway", -1.0, 6.5, 0.0, 2.4),
        slab("low-step", 2.8, 3.4, 0.1, 2.4, kind="step"),
    ]
    return make_level(
        level_id="level-01-basics",
        title="First Steps",
        order=1,
        difficulty=1,
        summary="A short flat runway with one low step: validate balance, contact and forward progress before adding gaps.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=6.0,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=20.0,
        par_time=9.0,
        tags=["flat", "step", "starter"],
    )


def level_02() -> dict:
    obstacles = [
        slab("runway", -1.0, 8.5, 0.0, 2.6),
        slab("raised-section", 3.0, 5.4, 0.18, 2.6, kind="step"),
    ]
    return make_level(
        level_id="level-02-step-up",
        title="Step Up, Step Down",
        order=2,
        difficulty=1,
        summary="Walk onto and off a raised 18 cm platform section without tripping.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.0,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=22.0,
        par_time=10.0,
        tags=["flat", "step"],
    )


def level_03() -> dict:
    obstacles = [
        slab("start", -1.0, 2.6, 0.0, 2.4),
        gap("gap-1", 2.6, 2.9, 2.4),
        slab("mid", 2.9, 4.8, 0.0, 1.9),
        gap("gap-2", 4.8, 5.1, 1.9),
        slab("end", 5.1, 7.4, 0.0, 1.7),
    ]
    return make_level(
        level_id="level-03-small-hops",
        title="Small Hops",
        order=3,
        difficulty=2,
        summary="Two 30 cm gaps: learn to step or hop across without losing balance.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=6.9,
        finish_radius=0.85,
        obstacles=obstacles,
        time_limit=24.0,
        par_time=11.0,
        tags=["gap", "hop"],
    )


def level_04() -> dict:
    obstacles = [
        slab("start", -1.0, 2.2, 0.0, 2.4),
        slab("beam", 2.2, 6.2, 0.0, 0.5, kind="rail"),
        slab("end", 6.2, 8.2, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-04-balance-beam",
        title="Balance Beam",
        order=4,
        difficulty=2,
        summary="Cross a 4 m long beam that is only 50 cm wide; lateral foot placement matters.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=7.5,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=26.0,
        par_time=12.0,
        tags=["balance", "narrow"],
    )


def level_05() -> dict:
    up = stairs("up", 2.2, count=5, rise=0.14, run=0.42, start_top=0.0, width=2.2)
    down = stairs(
        "down", 5.4, count=3, rise=0.14, run=0.42, start_top=0.7, width=2.2, ascending=False
    )
    obstacles = [
        slab("start", -1.0, 2.2, 0.0, 2.4),
        *up,
        slab("plateau", 4.3, 5.4, 0.7, 2.2),
        *down,
        slab("end", 6.66, 9.0, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-05-stairway",
        title="Stairway",
        order=5,
        difficulty=2,
        summary="Climb five 14 cm stairs, cross the plateau and descend three more.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.4,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=28.0,
        par_time=14.0,
        tags=["stairs", "elevation"],
    )


def level_06() -> dict:
    obstacles = [
        slab("runway", -1.0, 8.0, 0.0, 2.6),
        slab("hurdle-1", 2.5, 2.62, 0.2, 2.2, kind="wall"),
        slab("hurdle-2", 4.5, 4.62, 0.28, 2.2, kind="wall"),
        slab("hurdle-3", 6.5, 6.62, 0.36, 2.2, kind="wall"),
    ]
    return make_level(
        level_id="level-06-hurdles",
        title="Hurdles",
        order=6,
        difficulty=2,
        summary="Step over three thin hurdles of increasing height.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=7.5,
        finish_radius=0.85,
        obstacles=obstacles,
        time_limit=26.0,
        par_time=12.0,
        tags=["hurdle", "step"],
    )


def level_07() -> dict:
    obstacles = [
        slab("runway", -1.0, 8.0, 0.0, 2.8),
        slab("spikes-a", 2.6, 3.6, 0.06, 1.3, y=0.7, kind="spike_strip", hazard=True),
        slab("spikes-b", 4.8, 5.8, 0.06, 1.3, y=-0.7, kind="spike_strip", hazard=True),
    ]
    return make_level(
        level_id="level-07-spike-lane",
        title="Spike Lane",
        order=7,
        difficulty=2,
        summary="Two spike strips block one side of the lane: weave through the free side.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=7.4,
        finish_radius=0.85,
        obstacles=obstacles,
        time_limit=28.0,
        par_time=13.0,
        tags=["hazard", "weave"],
    )


def level_08() -> dict:
    obstacles = [
        slab("start", -1.0, 3.0, 0.0, 2.4),
        gap("gap-1", 3.0, 3.5, 2.4),
        slab("middle", 3.5, 5.2, 0.0, 2.2),
        gap("gap-2", 5.2, 5.7, 2.2),
        slab("end", 5.7, 8.2, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-08-gap-jump",
        title="Gap Jump",
        order=8,
        difficulty=3,
        summary="Two 50 cm gaps that need a real push-off rather than a single step.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=7.6,
        finish_radius=0.85,
        obstacles=obstacles,
        time_limit=28.0,
        par_time=13.0,
        tags=["gap", "jump"],
    )


def level_09() -> dict:
    obstacles = [
        slab("start", -1.0, 2.2, 0.0, 2.2),
        gap("gap-1", 2.2, 2.55, 1.2),
        box("stone-1", 2.9, 0.0, length=0.7, width=0.7, y=0.25),
        gap("gap-2", 3.25, 3.6, 0.7),
        box("stone-2", 3.95, 0.0, length=0.7, width=0.7, y=-0.25),
        gap("gap-3", 4.3, 4.65, 0.7),
        box("stone-3", 4.99, 0.0, length=0.7, width=0.7, y=0.25),
        gap("gap-4", 5.35, 5.7, 0.7),
        box("stone-4", 6.05, 0.0, length=0.7, width=0.7, y=-0.25),
        gap("gap-5", 6.4, 6.75, 0.7),
        slab("end", 6.75, 8.6, 0.0, 2.2),
    ]
    return make_level(
        level_id="level-09-stepping-stones",
        title="Stepping Stones",
        order=9,
        difficulty=3,
        summary="Four offset 70 cm stones with 35 cm gaps: precise foot placement plus lateral correction.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.0,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=32.0,
        par_time=15.0,
        tags=["precision", "gap", "lateral"],
    )


def level_10() -> dict:
    obstacles = [
        slab("start", -1.0, 2.6, 0.0, 2.4),
        slab("end", 5.4, 8.4, 0.0, 2.4),
        gap("gap-1", 2.6, 5.4, 2.2),
        box(
            "moving-bridge",
            4.0,
            0.0,
            length=1.4,
            width=1.5,
            kind="moving_platform",
            motion={"axis": [1.0, 0.0, 0.0], "amplitude": 0.55, "period_seconds": 4.0},
        ),
    ]
    return make_level(
        level_id="level-10-moving-bridge",
        title="Moving Bridge",
        order=10,
        difficulty=3,
        summary="A 2.8 m gap is bridged by a platform sliding back and forth: time the crossing.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=7.6,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=34.0,
        par_time=16.0,
        tags=["moving", "timing", "gap"],
    )


def level_11() -> dict:
    obstacles = [
        slab("start", -1.0, 2.4, 0.0, 2.4),
        slab("rise-1", 2.4, 2.9, 0.16, 2.2, kind="step"),
        slab("rise-2", 2.9, 3.4, 0.32, 2.2, kind="step"),
        slab("rise-3", 3.4, 3.9, 0.48, 2.2, kind="step"),
        slab("plateau", 3.9, 5.4, 0.5, 2.2),
        slab("drop-1", 5.4, 5.9, 0.32, 2.2, kind="step"),
        slab("drop-2", 5.9, 6.4, 0.16, 2.2, kind="step"),
        slab("end", 6.4, 8.6, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-11-up-and-over",
        title="Up and Over",
        order=11,
        difficulty=3,
        summary="Climb a 50 cm block in three coarse steps, cross it and drop down the far side.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.0,
        finish_radius=0.85,
        obstacles=obstacles,
        time_limit=30.0,
        par_time=14.0,
        tags=["stairs", "elevation"],
    )


def level_12() -> dict:
    obstacles = [
        slab("start", -1.0, 2.0, 0.0, 2.2),
        gap("gap-1", 2.0, 2.35, 0.8),
        slab("ledge-1", 2.35, 3.5, 0.0, 0.7),
        gap("gap-2", 3.5, 3.85, 0.7),
        slab("ledge-2", 3.85, 5.0, 0.0, 0.65),
        gap("gap-3", 5.0, 5.35, 0.6),
        slab("ledge-3", 5.35, 6.5, 0.0, 0.6, kind="rail"),
        gap("gap-4", 6.5, 6.85, 0.6),
        slab("end", 6.85, 8.6, 0.0, 2.0),
    ]
    return make_level(
        level_id="level-12-narrow-ledges",
        title="Narrow Ledges",
        order=12,
        difficulty=3,
        summary="Hop between ledges only 60-70 cm wide with 35 cm gaps.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.0,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=32.0,
        par_time=15.0,
        tags=["precision", "narrow", "gap"],
    )


def level_13() -> dict:
    obstacles = [
        slab("start", -1.0, 3.0, 0.0, 2.4),
        gap("gap-1", 3.0, 3.9, 2.2),
        slab("middle", 3.9, 6.0, 0.0, 2.0),
        gap("gap-2", 6.0, 6.8, 2.0),
        slab("end", 6.8, 9.2, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-13-broad-jump",
        title="Broad Jump",
        order=13,
        difficulty=4,
        summary="A 90 cm and an 80 cm gap: full extension jumps with a controlled landing.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.6,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=30.0,
        par_time=14.0,
        tags=["gap", "jump"],
    )


def level_14() -> dict:
    obstacles = [
        slab("start", -1.0, 2.0, 0.0, 2.4),
        gap("gap-1", 2.0, 2.3, 1.1),
        box("platform-1", 2.85, 0.0, length=1.1, width=1.1, y=0.65),
        gap("gap-2", 3.4, 3.7, 1.1),
        box("platform-2", 4.25, 0.0, length=1.1, width=1.1, y=-0.65),
        gap("gap-3", 4.8, 5.1, 1.1),
        box("platform-3", 5.65, 0.0, length=1.1, width=1.1, y=0.65),
        gap("gap-4", 6.2, 6.5, 1.1),
        box("platform-4", 7.05, 0.0, length=1.1, width=1.1, y=-0.65),
        gap("gap-5", 7.6, 7.9, 1.1),
        slab("end", 7.9, 10.0, 0.0, 2.2),
    ]
    return make_level(
        level_id="level-14-zigzag",
        title="Zigzag",
        order=14,
        difficulty=4,
        summary="Alternating left and right platforms with 40 cm gaps force turning jumps.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=9.4,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=34.0,
        par_time=17.0,
        tags=["lateral", "jump", "gap"],
    )


def level_15() -> dict:
    obstacles = [
        slab("start", -1.0, 2.8, 0.0, 2.4),
        gap("gap-1", 2.8, 3.55, 1.9),
        slab("landing", 3.55, 5.0, 0.0, 1.7),
        gap("gap-2", 5.0, 5.75, 1.7),
        slab("middle", 5.75, 7.4, 0.0, 1.6),
        gap("gap-3", 7.4, 8.1, 1.6),
        slab("end", 8.1, 10.4, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-15-double-gap",
        title="Double Gap",
        order=15,
        difficulty=4,
        summary="Three 70-75 cm gaps with short landings: chained jumps without resetting posture.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=9.8,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=34.0,
        par_time=16.0,
        tags=["gap", "jump", "chained"],
    )


def level_16() -> dict:
    obstacles = [
        slab("roof-1", -1.0, 2.6, 0.8, 2.2, kind="rooftop", bottom=-0.4),
        gap("gap-1", 2.6, 3.15, 2.2),
        slab("roof-2", 3.15, 4.7, 0.8, 1.9, kind="rooftop", bottom=-0.4),
        gap("gap-2", 4.7, 5.25, 1.9),
        slab("roof-3", 5.25, 6.6, 0.8, 1.7, kind="rooftop", bottom=-0.4),
        gap("gap-3", 6.6, 7.15, 1.7),
        slab("roof-4", 7.15, 9.4, 0.8, 2.0, kind="rooftop", bottom=-0.4),
    ]
    return make_level(
        level_id="level-16-rooftops",
        title="Rooftops",
        order=16,
        difficulty=4,
        summary="Elevated platforms 80 cm above the void: a missed landing is a real fall.",
        spawn_x=0.0,
        spawn_top=0.8,
        finish_x=8.8,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=36.0,
        par_time=17.0,
        tags=["height", "gap", "fall-risk"],
    )


def level_17() -> dict:
    obstacles = [
        slab("runway", -1.0, 10.0, 0.0, 3.0),
        slab("spikes-1", 2.5, 3.0, 0.07, 2.8, kind="spike_strip", hazard=True),
        slab("spikes-2", 5.0, 5.5, 0.07, 2.8, kind="spike_strip", hazard=True),
        slab("spikes-3", 7.5, 8.0, 0.07, 2.8, kind="spike_strip", hazard=True),
    ]
    return make_level(
        level_id="level-17-spike-gauntlet",
        title="Spike Gauntlet",
        order=17,
        difficulty=4,
        summary="Three full-width 50 cm spike strips: jump each one and keep enough momentum for the next.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=9.4,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=34.0,
        par_time=16.0,
        tags=["hazard", "jump"],
    )


def level_18() -> dict:
    obstacles = [
        slab("start", -1.0, 2.2, 0.0, 2.2),
        slab("rise-1", 2.2, 2.7, 0.18, 2.0, kind="step"),
        slab("rise-2", 2.7, 3.2, 0.36, 2.0, kind="step"),
        slab("rail", 3.2, 6.6, 0.5, 0.3, kind="rail"),
        slab("drop-1", 6.6, 7.1, 0.3, 2.0, kind="step"),
        slab("drop-2", 7.1, 7.6, 0.12, 2.0, kind="step"),
        slab("end", 7.6, 9.6, 0.0, 2.2),
    ]
    return make_level(
        level_id="level-18-rail-run",
        title="Rail Run",
        order=18,
        difficulty=4,
        summary="Climb 36 cm, balance along a 30 cm wide elevated rail and descend safely.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=9.0,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=38.0,
        par_time=18.0,
        tags=["narrow", "height", "balance"],
    )


def level_19() -> dict:
    obstacles = [
        slab("start", -1.0, 2.4, 0.0, 2.4),
        gap("gap-1", 2.4, 2.8, 2.0),
        slab("tier-1", 2.8, 4.0, 0.35, 1.9),
        gap("gap-2", 4.0, 4.4, 1.8),
        slab("tier-2", 4.4, 5.6, 0.7, 1.7),
        gap("gap-3", 5.6, 6.0, 1.6),
        slab("tier-3", 6.0, 7.2, 1.05, 1.6),
        gap("gap-4", 7.2, 7.6, 1.5),
        slab("tier-4", 7.6, 9.0, 1.4, 1.8, kind="rooftop", bottom=-0.4),
    ]
    return make_level(
        level_id="level-19-high-rise",
        title="High Rise",
        order=19,
        difficulty=5,
        summary="Four ascending tiers 35 cm apart with gaps: jump up, land short, repeat.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.4,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=42.0,
        par_time=20.0,
        tags=["height", "jump", "elevation"],
    )


def level_20() -> dict:
    obstacles = [
        slab("start", -1.0, 2.2, 0.0, 2.6),
        slab("step-up", 2.2, 2.8, 0.15, 2.4, kind="step"),
        gap("gap-1", 2.8, 3.15, 2.4),
        slab("landing", 3.15, 4.3, 0.15, 2.4),
        slab("step-down", 4.3, 4.9, 0.0, 2.6),
        gap("gap-2", 4.9, 5.4, 2.4),
        slab("spike-run", 5.4, 7.0, 0.0, 2.6),
        slab("spikes", 6.0, 6.5, 0.07, 2.4, kind="spike_strip", hazard=True),
        gap("gap-3", 7.0, 7.5, 2.0),
        slab("beam", 7.5, 9.5, 0.0, 0.35, kind="rail"),
        gap("gap-4", 9.5, 10.5, 1.6),
        box(
            "moving-platform",
            11.0,
            0.0,
            length=1.4,
            width=1.4,
            kind="moving_platform",
            motion={"axis": [0.0, 1.0, 0.0], "amplitude": 0.5, "period_seconds": 3.6},
        ),
        gap("gap-5", 11.55, 12.5, 1.6),
        slab("finish-zone", 12.5, 14.6, 0.0, 2.4),
    ]
    return make_level(
        level_id="level-20-final-gauntlet",
        title="Final Gauntlet",
        order=20,
        difficulty=5,
        summary="Steps, gaps, spikes, a narrow beam and a sliding platform chained into one course.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=14.0,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=50.0,
        par_time=26.0,
        tags=["mixed", "hazard", "moving", "narrow"],
    )


def level_21_heldout() -> dict:
    obstacles = [
        slab("start", -1.0, 2.0, 0.0, 2.4),
        gap("gap-1", 2.0, 2.45, 2.0),
        slab("stone-a", 2.45, 3.35, 0.2, 0.9, kind="rooftop", bottom=-0.4),
        gap("gap-2", 3.35, 3.8, 0.9),
        slab("stone-b", 3.8, 4.7, 0.2, 0.9, kind="rooftop", bottom=-0.4),
        gap("gap-3", 4.7, 5.2, 0.9),
        slab("shelf", 5.2, 6.8, 0.45, 1.2),
        gap("gap-4", 6.8, 7.3, 1.2),
        slab("spike-run", 7.3, 9.3, 0.45, 2.2),
        slab("spikes", 8.0, 8.4, 0.52, 2.0, kind="spike_strip", hazard=True),
        gap("gap-5", 9.3, 9.8, 1.8),
        slab("end", 9.8, 11.8, 0.45, 2.0),
    ]
    return make_level(
        level_id="level-21-heldout-course",
        title="Held-out Course",
        order=21,
        difficulty=5,
        summary="Unseen generalisation course: raised stones, a shelf and a spike run at 45 cm height.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=11.2,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=44.0,
        par_time=22.0,
        tags=["held-out", "mixed"],
        split="held-out",
    )


def level_22_heldout() -> dict:
    obstacles = [
        slab("start", -1.0, 1.8, 0.0, 2.0),
        gap("gap-1", 1.8, 2.2, 1.0),
        slab("pad-1", 2.2, 3.05, 0.0, 0.8, kind="rail"),
        gap("gap-2", 3.05, 3.45, 0.8),
        slab("pad-2", 3.45, 4.3, 0.12, 0.8, kind="rail"),
        gap("gap-3", 4.3, 4.7, 0.8),
        slab("pad-3", 4.7, 5.55, 0.24, 0.8, kind="rail"),
        gap("gap-4", 5.55, 5.95, 0.8),
        slab("pad-4", 5.95, 6.8, 0.36, 0.8, kind="rail"),
        gap("gap-5", 6.8, 7.2, 1.0),
        slab("end", 7.2, 9.4, 0.0, 2.0),
    ]
    return make_level(
        level_id="level-22-heldout-ledges",
        title="Held-out Ledges",
        order=22,
        difficulty=5,
        summary="Unseen generalisation course: four rising 80 cm pads with gaps, then a drop.",
        spawn_x=0.0,
        spawn_top=0.0,
        finish_x=8.8,
        finish_radius=0.9,
        obstacles=obstacles,
        time_limit=40.0,
        par_time=20.0,
        tags=["held-out", "precision", "height"],
        split="held-out",
    )


BUILDERS = [
    level_01,
    level_02,
    level_03,
    level_04,
    level_05,
    level_06,
    level_07,
    level_08,
    level_09,
    level_10,
    level_11,
    level_12,
    level_13,
    level_14,
    level_15,
    level_16,
    level_17,
    level_18,
    level_19,
    level_20,
    level_21_heldout,
    level_22_heldout,
]


def render(level: dict) -> str:
    return json.dumps(level, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify files without writing")
    parser.add_argument("--out", type=Path, default=LEVELS_DIR)
    args = parser.parse_args(argv)

    levels = [builder() for builder in BUILDERS]
    args.out.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for level in levels:
        path = args.out / f"{level['id']}.json"
        text = render(level)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print(f"stale level manifests: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"{len(levels)} level manifests are up to date")
        return 0
    print(f"wrote {len(levels)} level manifests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
