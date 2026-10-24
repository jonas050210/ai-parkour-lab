"""Turn a validated :class:`~parkour_lab.levels.Level` into a complete MJCF scene.

The course builder is the single place where level data meets physics.  It inserts the
humanoid from :mod:`parkour_lab.humanoid` and one box geom per solid obstacle into one
MJCF document, names every geom so runtime code can map contacts back to level semantics,
and marks hazard and moving platforms explicitly.

There is deliberately **no infinite ground plane under the course**: every level is a
set of platforms above the void.  Falling off a platform is therefore a real fall that the
environment detects, which is what gives rooftops and gaps their consequence. A
non-colliding visual floor exists only to provide depth cues in the native viewer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .humanoid import (
    HUMANOID_SUBDEFAULTS,
    HumanoidSpec,
    default_humanoid,
    humanoid_actuator_xml,
    humanoid_worldbody_xml,
)
from .levels import Level, Obstacle

_ROBOT_BODIES = (
    "pelvis",
    "thigh_l",
    "shin_l",
    "foot_l",
    "thigh_r",
    "shin_r",
    "foot_r",
    "torso",
    "head",
    "upperarm_l",
    "forearm_l",
    "upperarm_r",
    "forearm_r",
)


@dataclass(frozen=True, slots=True)
class MovingPlatform:
    """Runtime description of a mocap-driven platform."""

    obstacle_id: str
    geom_name: str
    body_name: str
    base_position: tuple[float, float, float]
    axis: tuple[float, float, float]
    amplitude: float
    period: float

    def position_at(self, time_seconds: float) -> tuple[float, float, float]:
        phase = 2.0 * math.pi * (time_seconds / self.period)
        offset = self.amplitude * math.sin(phase)
        return (
            self.base_position[0] + self.axis[0] * offset,
            self.base_position[1] + self.axis[1] * offset,
            self.base_position[2] + self.axis[2] * offset,
        )


@dataclass(slots=True)
class CourseScene:
    """Compiled scene description plus the metadata runtime code needs."""

    level: Level
    xml: str
    humanoid: HumanoidSpec
    geom_to_obstacle: dict[str, str] = field(default_factory=dict)
    hazard_geoms: dict[str, str] = field(default_factory=dict)
    solid_geoms: dict[str, str] = field(default_factory=dict)
    moving_platforms: tuple[MovingPlatform, ...] = ()
    robot_bodies: tuple[str, ...] = _ROBOT_BODIES

    @property
    def hazard_geom_names(self) -> tuple[str, ...]:
        return tuple(self.hazard_geoms)


def _vec(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _geom_class(obstacle: Obstacle) -> str:
    if obstacle.hazard:
        return "parkour_hazard"
    if obstacle.kind == "rail":
        return "parkour_rail"
    return "parkour_obstacle"


def build_course(level: Level, humanoid: HumanoidSpec | None = None) -> CourseScene:
    """Assemble the MJCF scene for ``level``.

    Raises :class:`ValueError` if the level has no solid geometry (already checked by the
    loader, but the builder does not assume the caller used the loader).
    """

    spec = humanoid if humanoid is not None else default_humanoid()
    if not level.solid_obstacles:
        raise ValueError(f"level {level.id!r} has no solid obstacles")

    static_lines: list[str] = []
    moving_lines: list[str] = []
    moving_platforms: list[MovingPlatform] = []
    geom_to_obstacle: dict[str, str] = {}
    solid_geoms: dict[str, str] = {}
    hazard_geoms: dict[str, str] = {}

    static_lines.append(
        '    <geom name="visual_floor" type="plane" pos="0 0 -2.5" size="40 40 0.05" '
        'contype="0" conaffinity="0" material="grid_mat" rgba="0.16 0.18 0.22 1"/>'
    )
    static_lines.append(
        f'    <geom name="start_marker" type="cylinder" pos="{_vec((level.spawn_position.x, level.spawn_position.y, level.spawn_position.z - 0.9))}" '
        f'size="0.45 0.02" contype="0" conaffinity="0" rgba="0.25 0.75 0.95 0.35"/>'
    )
    static_lines.append(
        f'    <geom name="finish_marker" type="cylinder" pos="{_vec((level.finish_position.x, level.finish_position.y, level.finish_position.z - 0.9))}" '
        f'size="{level.finish_radius:.4g} 0.03" contype="0" conaffinity="0" '
        f'rgba="0.25 0.95 0.45 0.35"/>'
    )
    static_lines.append(
        f'    <site name="finish_site" pos="{_vec((level.finish_position.x, level.finish_position.y, level.finish_position.z + 0.4))}" '
        f'size="0.08" rgba="0.2 1 0.4 1"/>'
    )

    for obstacle in level.obstacles:
        if not obstacle.solid:
            continue
        geom_name = f"obs_{obstacle.id}"
        if obstacle.motion is not None:
            body_name = f"mover_{obstacle.id}"
            moving_lines.append(
                f'    <body name="{body_name}" mocap="true" pos="{_vec(obstacle.position.as_tuple())}">'
            )
            moving_lines.append(
                f'      <geom name="{geom_name}" class="{_geom_class(obstacle)}" type="box" '
                f'pos="0 0 0" size="{_vec(obstacle.size.as_tuple())}" '
                f'euler="0 0 {obstacle.rotation_degrees:.6g}"/>'
            )
            moving_lines.append("    </body>")
            moving_platforms.append(
                MovingPlatform(
                    obstacle_id=obstacle.id,
                    geom_name=geom_name,
                    body_name=body_name,
                    base_position=obstacle.position.as_tuple(),
                    axis=obstacle.motion.axis.as_tuple(),
                    amplitude=obstacle.motion.amplitude,
                    period=obstacle.motion.period_seconds,
                )
            )
        else:
            static_lines.append(
                f'    <geom name="{geom_name}" class="{_geom_class(obstacle)}" type="box" '
                f'pos="{_vec(obstacle.position.as_tuple())}" '
                f'size="{_vec(obstacle.size.as_tuple())}" '
                f'euler="0 0 {obstacle.rotation_degrees:.6g}"/>'
            )
        geom_to_obstacle[geom_name] = obstacle.id
        solid_geoms[geom_name] = obstacle.id
        if obstacle.hazard:
            hazard_geoms[geom_name] = obstacle.id

    robot_body = humanoid_worldbody_xml(spec, indent="      ")
    actuator_block = humanoid_actuator_xml(spec, indent="    ")
    parts = [
        '<mujoco model="parkour-lab">',
        '  <compiler angle="degree" autolimits="true" inertiafromgeom="true"/>',
        f'  <option timestep="{spec.timestep:.6g}" integrator="implicitfast" '
        f'gravity="0 0 -9.81" iterations="60" solver="Newton"/>',
        "  <visual>",
        '    <global offwidth="1280" offheight="720" ellipsoidinertia="true"/>',
        '    <quality shadowsize="4096"/>',
        '    <headlight ambient="0.35 0.35 0.40" diffuse="0.6 0.6 0.6"/>',
        "  </visual>",
        '  <asset>',
        '    <texture name="grid_tex" type="2d" builtin="checker" width="256" height="256" '
        'rgb1="0.18 0.20 0.24" rgb2="0.24 0.26 0.30"/>',
        '    <material name="grid_mat" texture="grid_tex" texrepeat="24 24" reflectance="0.05"/>',
        "  </asset>",
        HUMANOID_SUBDEFAULTS.rstrip(),
        "  <worldbody>",
        *static_lines,
        *moving_lines,
        robot_body,
        "  </worldbody>",
        actuator_block,
        "  <sensor>",
        '    <framepos name="pelvis_pos" objtype="body" objname="pelvis"/>',
        '    <framequat name="pelvis_quat" objtype="body" objname="pelvis"/>',
        '    <framelinvel name="pelvis_linvel" objtype="body" objname="pelvis"/>',
        '    <frameangvel name="pelvis_angvel" objtype="body" objname="pelvis"/>',
        "  </sensor>",
        "</mujoco>",
    ]
    xml = "\n".join(parts) + "\n"

    return CourseScene(
        level=level,
        xml=xml,
        humanoid=spec,
        geom_to_obstacle=geom_to_obstacle,
        hazard_geoms=hazard_geoms,
        solid_geoms=solid_geoms,
        moving_platforms=tuple(moving_platforms),
    )


__all__ = ["CourseScene", "MovingPlatform", "build_course"]
