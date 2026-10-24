"""Procedural MJCF humanoid used by every parkour level.

The body is a complete articulated humanoid (25 degrees of freedom: six free-root
coordinates plus nineteen actuated joints).  It is a *game/physics* humanoid, not an
anatomical model: segment masses and lengths are chosen for stable, learnable
locomotion, and the default actuation is bounded position-servo control with explicit
force limits.

Two actuation modes are supported:

``position`` (default)
    Every joint is driven by a MuJoCo ``position`` actuator with stiffness/damping
    gains and a hard force range.  The policy emits normalised targets in ``[-1, 1]``
    that are mapped onto the joint range.  This is bounded, stable and easy to debug.

``muscle``
    Experimental, muscle-*inspired* activation dynamics: each joint receives a
    first-order activation state (time constant ``tau``) that low-pass filters the
    normalised torque command, with an explicit force ceiling.  It is a research hook
    for comparing activation dynamics with pure position servos; it is **not** an
    anatomical Hill-type musculoskeletal model and makes no biomechanical claim.

Everything here is deterministic XML generation with no I/O, so the model can be
inspected, exported and unit-tested without a physics world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

ActuationMode = Literal["position", "muscle"]

# --------------------------------------------------------------------------------------
# Geometry. All lengths in metres, masses in kilograms, angles in degrees. The frame is
# MuJoCo's: +x forward, +y left, +z up. Segment lengths sum to the standing hip height so
# the default pose stands on the ground with a straight leg.
# --------------------------------------------------------------------------------------

PELVIS_HEIGHT = 0.92
THIGH_LENGTH = 0.40
SHIN_LENGTH = 0.38
FOOT_HEIGHT = 0.07

TOTAL_STANDING_HEIGHT = 1.72  # pelvis height + torso + head, used by tests/docs


@dataclass(frozen=True, slots=True)
class JointSpec:
    """One actuated scalar joint."""

    name: str
    body: str
    pos: tuple[float, float, float]
    axis: tuple[float, float, int]
    lower: float
    upper: float
    kp: float
    kv: float
    force_limit: float
    group: str
    default: float = 0.0
    damping: float = 1.5
    armature: float = 0.01

    @property
    def range(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.upper + self.lower)


@dataclass(frozen=True, slots=True)
class GeomSpec:
    """Visual+collision primitive attached to a body."""

    body: str
    kind: Literal["box", "capsule", "sphere", "cylinder"]
    pos: tuple[float, float, float]
    size: tuple[float, ...]
    mass: float
    rgba: tuple[float, float, float, float]
    name: str | None = None


@dataclass(frozen=True, slots=True)
class BodySpec:
    name: str
    parent: str | None
    pos: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class HumanoidSpec:
    """Complete, validated description of the simulated humanoid."""

    model_name: str = "parkour-humanoid"
    bodies: tuple[BodySpec, ...] = field(default_factory=tuple)
    joints: tuple[JointSpec, ...] = field(default_factory=tuple)
    geoms: tuple[GeomSpec, ...] = field(default_factory=tuple)
    sites: tuple[tuple[str, str, tuple[float, float, float]], ...] = field(
        default_factory=tuple
    )
    actuation_mode: ActuationMode = "position"
    muscle_tau: float = 0.045
    timestep: float = 0.002
    control_decimation: int = 5

    # -- derived inventory -----------------------------------------------------------------
    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def action_dim(self) -> int:
        return len(self.joints)

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        return tuple(j.lower for j in self.joints)

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        return tuple(j.upper for j in self.joints)

    @property
    def default_joint_positions(self) -> tuple[float, ...]:
        return tuple(j.default for j in self.joints)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.control_decimation

    @property
    def total_mass(self) -> float:
        # Bodies only carry mass through their geoms in this model (no explicit body mass).
        return math.fsum(g.mass for g in self.geoms)

    def with_actuation(self, mode: ActuationMode, *, muscle_tau: float | None = None) -> "HumanoidSpec":
        """Return a copy using another actuation mode (used by the muscle experiment)."""

        if mode not in ("position", "muscle"):
            raise ValueError(f"unknown actuation mode: {mode!r}")
        return replace(
            self,
            actuation_mode=mode,
            muscle_tau=self.muscle_tau if muscle_tau is None else float(muscle_tau),
        )

    def by_group(self) -> dict[str, tuple[JointSpec, ...]]:
        groups: dict[str, list[JointSpec]] = {}
        for joint in self.joints:
            groups.setdefault(joint.group, []).append(joint)
        return {key: tuple(value) for key, value in groups.items()}


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------

_SKIN = (0.78, 0.60, 0.48, 1.0)
_SHIRT = (0.20, 0.45, 0.75, 1.0)
_PANTS = (0.22, 0.24, 0.30, 1.0)
_SHOE = (0.10, 0.10, 0.12, 1.0)


def default_humanoid() -> HumanoidSpec:
    """Build the standard 25-DoF humanoid description."""

    bodies = [
        BodySpec("pelvis", None, (0.0, 0.0, PELVIS_HEIGHT)),
        # legs (left = +y)
        BodySpec("thigh_l", "pelvis", (0.0, 0.095, -0.075)),
        BodySpec("shin_l", "thigh_l", (0.0, 0.0, -THIGH_LENGTH)),
        BodySpec("foot_l", "shin_l", (0.0, 0.0, -SHIN_LENGTH)),
        BodySpec("thigh_r", "pelvis", (0.0, -0.095, -0.075)),
        BodySpec("shin_r", "thigh_r", (0.0, 0.0, -THIGH_LENGTH)),
        BodySpec("foot_r", "shin_r", (0.0, 0.0, -SHIN_LENGTH)),
        # torso and head
        BodySpec("torso", "pelvis", (0.0, 0.0, 0.09)),
        BodySpec("head", "torso", (0.0, 0.0, 0.43)),
        # arms
        BodySpec("upperarm_l", "torso", (0.0, 0.205, 0.34)),
        BodySpec("forearm_l", "upperarm_l", (0.0, 0.0, -0.26)),
        BodySpec("upperarm_r", "torso", (0.0, -0.205, 0.34)),
        BodySpec("forearm_r", "upperarm_r", (0.0, 0.0, -0.26)),
    ]

    joints = [
        # ---- hip: three hinges give a limited ball joint without needing ball-quaternion
        # handling in the actuator interface.
        JointSpec("hip_pitch_l", "thigh_l", (0, 0, 0), (0, 1, 0), -115.0, 45.0,
                  kp=320, kv=14, force_limit=260, group="leg", default=0.0),
        JointSpec("hip_roll_l", "thigh_l", (0, 0, 0), (1, 0, 0), -22.0, 48.0,
                  kp=260, kv=12, force_limit=220, group="leg", default=-2.0),
        JointSpec("hip_yaw_l", "thigh_l", (0, 0, 0), (0, 0, 1), -40.0, 40.0,
                  kp=140, kv=6, force_limit=120, group="leg"),
        JointSpec("knee_l", "shin_l", (0, 0, 0), (0, 1, 0), 2.0, 140.0,
                  kp=340, kv=14, force_limit=280, group="leg", default=2.0),
        JointSpec("ankle_l", "foot_l", (0, 0, 0), (0, 1, 0), -48.0, 42.0,
                  kp=260, kv=7, force_limit=150, group="leg", default=-2.0),
        JointSpec("hip_pitch_r", "thigh_r", (0, 0, 0), (0, 1, 0), -115.0, 45.0,
                  kp=320, kv=14, force_limit=260, group="leg", default=0.0),
        JointSpec("hip_roll_r", "thigh_r", (0, 0, 0), (1, 0, 0), -48.0, 22.0,
                  kp=260, kv=12, force_limit=220, group="leg", default=2.0),
        JointSpec("hip_yaw_r", "thigh_r", (0, 0, 0), (0, 0, 1), -40.0, 40.0,
                  kp=140, kv=6, force_limit=120, group="leg"),
        JointSpec("knee_r", "shin_r", (0, 0, 0), (0, 1, 0), 2.0, 140.0,
                  kp=340, kv=14, force_limit=280, group="leg", default=2.0),
        JointSpec("ankle_r", "foot_r", (0, 0, 0), (0, 1, 0), -48.0, 42.0,
                  kp=260, kv=7, force_limit=150, group="leg", default=-2.0),
        # ---- spine / neck
        JointSpec("spine_pitch", "torso", (0, 0, 0), (0, 1, 0), -32.0, 42.0,
                  kp=300, kv=16, force_limit=240, group="torso"),
        JointSpec("spine_twist", "torso", (0, 0, 0), (0, 0, 1), -50.0, 50.0,
                  kp=180, kv=8, force_limit=150, group="torso"),
        JointSpec("neck_pitch", "head", (0, 0, 0), (0, 1, 0), -42.0, 42.0,
                  kp=22, kv=1.5, force_limit=25, group="torso"),
        # ---- arms
        JointSpec("shoulder_pitch_l", "upperarm_l", (0, 0, 0), (0, 1, 0), -165.0, 75.0,
                  kp=75, kv=4, force_limit=70, group="arm", default=6.0),
        JointSpec("shoulder_roll_l", "upperarm_l", (0, 0, 0), (1, 0, 0), -35.0, 155.0,
                  kp=60, kv=3, force_limit=60, group="arm", default=9.0),
        JointSpec("elbow_l", "forearm_l", (0, 0, 0), (0, 1, 0), -140.0, 4.0,
                  kp=45, kv=2.5, force_limit=45, group="arm", default=-18.0),
        JointSpec("shoulder_pitch_r", "upperarm_r", (0, 0, 0), (0, 1, 0), -165.0, 75.0,
                  kp=75, kv=4, force_limit=70, group="arm", default=6.0),
        JointSpec("shoulder_roll_r", "upperarm_r", (0, 0, 0), (1, 0, 0), -155.0, 35.0,
                  kp=60, kv=3, force_limit=60, group="arm", default=-9.0),
        JointSpec("elbow_r", "forearm_r", (0, 0, 0), (0, 1, 0), -140.0, 4.0,
                  kp=45, kv=2.5, force_limit=45, group="arm", default=-18.0),
    ]

    geoms = [
        # pelvis
        GeomSpec("pelvis", "box", (0, 0, 0), (0.135, 0.105, 0.085), 8.0, _PANTS, "pelvis_geom"),
        # legs
        GeomSpec("thigh_l", "capsule", (0, 0, -0.20), (0.062, THIGH_LENGTH / 2), 4.5, _PANTS, "thigh_l_geom"),
        GeomSpec("shin_l", "capsule", (0, 0, -0.19), (0.050, SHIN_LENGTH / 2), 3.2, _SKIN, "shin_l_geom"),
        GeomSpec("foot_l", "box", (0.0, 0, -0.035), (0.13, 0.055, 0.035), 1.1, _SHOE, "foot_l_geom"),
        GeomSpec("thigh_r", "capsule", (0, 0, -0.20), (0.062, THIGH_LENGTH / 2), 4.5, _PANTS, "thigh_r_geom"),
        GeomSpec("shin_r", "capsule", (0, 0, -0.19), (0.050, SHIN_LENGTH / 2), 3.2, _SKIN, "shin_r_geom"),
        GeomSpec("foot_r", "box", (0.0, 0, -0.035), (0.13, 0.055, 0.035), 1.1, _SHOE, "foot_r_geom"),
        # torso / head
        GeomSpec("torso", "box", (0, 0, 0.22), (0.165, 0.11, 0.215), 11.0, _SHIRT, "torso_geom"),
        GeomSpec("head", "sphere", (0, 0, 0.115), (0.105,), 3.6, _SKIN, "head_geom"),
        # arms
        GeomSpec("upperarm_l", "capsule", (0, 0, -0.13), (0.045, 0.13), 1.9, _SHIRT, "upperarm_l_geom"),
        GeomSpec("forearm_l", "capsule", (0, 0, -0.12), (0.037, 0.12), 1.3, _SKIN, "forearm_l_geom"),
        GeomSpec("upperarm_r", "capsule", (0, 0, -0.13), (0.045, 0.13), 1.9, _SHIRT, "upperarm_r_geom"),
        GeomSpec("forearm_r", "capsule", (0, 0, -0.12), (0.037, 0.12), 1.3, _SKIN, "forearm_r_geom"),
    ]

    sites = (
        ("imu", "pelvis", (0.0, 0.0, 0.0)),
        ("head_site", "head", (0.0, 0.0, 0.0)),
        ("foot_l_site", "foot_l", (0.0, 0.0, -0.035)),
        ("foot_r_site", "foot_r", (0.0, 0.0, -0.035)),
        ("hand_l_site", "forearm_l", (0.0, 0.0, -0.24)),
        ("hand_r_site", "forearm_r", (0.0, 0.0, -0.24)),
    )

    return HumanoidSpec(
        bodies=tuple(bodies),
        joints=tuple(joints),
        geoms=tuple(geoms),
        sites=sites,
    )


def _fmt(value: float) -> str:
    """Compact, deterministic number formatting for XML."""

    if value == 0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(float(v)) for v in values)


def humanoid_worldbody_xml(spec: HumanoidSpec, *, indent: str = "    ") -> str:
    """Render the articulated body tree (with the free root joint) as MJCF."""

    body_lines: list[str] = []
    bodies_by_parent: dict[str | None, list[BodySpec]] = {}
    for body in spec.bodies:
        bodies_by_parent.setdefault(body.parent, []).append(body)

    joints_by_body: dict[str, list[JointSpec]] = {}
    for joint in spec.joints:
        joints_by_body.setdefault(joint.body, []).append(joint)
    geoms_by_body: dict[str, list[GeomSpec]] = {}
    for geom in spec.geoms:
        geoms_by_body.setdefault(geom.body, []).append(geom)
    sites_by_body: dict[str, list[tuple[str, str, tuple[float, float, float]]]] = {}
    for name, body, pos in spec.sites:
        sites_by_body.setdefault(body, []).append((name, body, pos))

    def geom_line(geom: GeomSpec, pad: str) -> str:
        name_attr = f' name="{geom.name}"' if geom.name else ""
        return (
            f'{pad}<geom{name_attr} type="{geom.kind}" pos="{_vec(geom.pos)}" '
            f'size="{_vec(geom.size)}" mass="{_fmt(geom.mass)}" rgba="{_vec(geom.rgba)}" '
            f'friction="1.0 0.02 0.002" condim="3" solref="0.004 1"/>'
        )

    def emit(body: BodySpec, depth: int) -> None:
        pad = indent + "  " * depth
        body_lines.append(f'{pad}<body name="{body.name}" pos="{_vec(body.pos)}">')
        inner = pad + "  "
        for joint in joints_by_body.get(body.name, []):
            body_lines.append(
                f'{inner}<joint name="{joint.name}" type="hinge" axis="{_vec(joint.axis)}" '
                f'range="{_fmt(joint.lower)} {_fmt(joint.upper)}" '
                f'damping="{_fmt(joint.damping)}" armature="{_fmt(joint.armature)}" '
                f'limited="true"/>'
            )
        for geom in geoms_by_body.get(body.name, []):
            body_lines.append(geom_line(geom, inner))
        for name, _body, pos in sites_by_body.get(body.name, []):
            body_lines.append(f'{inner}<site name="{name}" pos="{_vec(pos)}" size="0.02"/>')
        for child in bodies_by_parent.get(body.name, []):
            emit(child, depth + 1)
        body_lines.append(f"{pad}</body>")

    roots = bodies_by_parent.get(None, [])
    if len(roots) != 1:
        raise ValueError("humanoid must have exactly one root body")
    root = roots[0]
    body_lines.append(f'{indent}<body name="{root.name}" pos="{_vec(root.pos)}">')
    body_lines.append(f'{indent}  <freejoint name="root"/>')
    for geom in geoms_by_body.get(root.name, []):
        body_lines.append(geom_line(geom, indent + "  "))
    for name, _body, pos in sites_by_body.get(root.name, []):
        body_lines.append(f'{indent}  <site name="{name}" pos="{_vec(pos)}" size="0.02"/>')
    for child in bodies_by_parent.get(root.name, []):
        emit(child, 1)
    body_lines.append(f"{indent}</body>")
    return "\n".join(body_lines)


def humanoid_actuator_xml(spec: HumanoidSpec, *, indent: str = "    ") -> str:
    """Render the actuator block matching :func:`humanoid_worldbody_xml`."""

    inner = indent + "  "
    lines: list[str] = [f"{indent}<actuator>"]
    for joint in spec.joints:
        if spec.actuation_mode == "position":
            # Joint ranges in the XML are degrees, but a position actuator's control (and
            # therefore ctrlrange) is in radians, matching qpos.
            ctrl_low = math.radians(joint.lower)
            ctrl_high = math.radians(joint.upper)
            lines.append(
                f'{inner}<position name="motor_{joint.name}" joint="{joint.name}" '
                f'kp="{_fmt(joint.kp)}" kv="{_fmt(joint.kv)}" '
                f'forcerange="{_fmt(-joint.force_limit)} {_fmt(joint.force_limit)}" '
                f'ctrllimited="true" ctrlrange="{_fmt(ctrl_low)} {_fmt(ctrl_high)}"/>'
            )
        else:
            # Normalised, activation-filtered torque command. ctrl in [-1, 1] is scaled by
            # `gear` to a force and clipped by `forcerange`; `dyntype=filter` adds the
            # first-order activation lag tau. This is muscle-inspired, not anatomical.
            lines.append(
                f'{inner}<general name="motor_{joint.name}" joint="{joint.name}" '
                f'dyntype="filter" dynprm="{_fmt(spec.muscle_tau)}" gaintype="fixed" gainprm="1" '
                f'biastype="none" gear="{_fmt(joint.force_limit)}" '
                f'forcerange="{_fmt(-joint.force_limit)} {_fmt(joint.force_limit)}" '
                f'ctrllimited="true" ctrlrange="-1 1"/>'
            )
    lines.append(f"{indent}</actuator>")
    return "\n".join(lines)


def humanoid_xml(spec: HumanoidSpec, *, indent: str = "      ") -> str:
    """Standalone MJCF fragment: body tree plus actuator block."""

    return humanoid_worldbody_xml(spec, indent=indent) + "\n" + humanoid_actuator_xml(spec, indent=indent)


# Kept separate so that the XML assembled by the course builder stays readable.
HUMANOID_SUBDEFAULTS = """
    <default>
      <default class="parkour_obstacle">
        <geom friction="1.1 0.02 0.002" condim="3" solref="0.004 1" rgba="0.55 0.57 0.60 1"/>
      </default>
      <default class="parkour_hazard">
        <geom friction="1.0 0.02 0.002" condim="3" solref="0.004 1" rgba="0.85 0.20 0.18 1"/>
      </default>
      <default class="parkour_rail">
        <geom friction="0.9 0.02 0.002" condim="3" solref="0.004 1" rgba="0.95 0.72 0.20 1"/>
      </default>
    </default>
"""


__all__ = [
    "ActuationMode",
    "BodySpec",
    "GeomSpec",
    "HumanoidSpec",
    "JointSpec",
    "PELVIS_HEIGHT",
    "TOTAL_STANDING_HEIGHT",
    "default_humanoid",
    "humanoid_actuator_xml",
    "humanoid_worldbody_xml",
    "humanoid_xml",
]
