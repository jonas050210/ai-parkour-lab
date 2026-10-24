"""Single-world MuJoCo parkour environment.

This is the physics and task reference used by training, evaluation and replay.  It is a
Gymnasium-shaped environment (``reset`` -> ``(obs, info)``, ``step`` ->
``(obs, reward, terminated, truncated, info)``) with an explicit, documented observation
and action layout.

Action layout
-------------
19 continuous values in ``[-1, 1]`` mapped onto the 19 actuated joint targets around the
standing pose (``action_scale`` controls the range).  In ``position`` mode each target is
clamped to the joint's mechanical range and executed by a bounded position servo; in
``muscle`` mode the numbers are normalised torque commands through first-order activation
dynamics with a hard force ceiling.

Observation layout (``OBSERVATION_NAMES``, 90 values)
-----------------------------------------------------
root (12)        support-relative pelvis height, up-vector z, body-frame up vector (3),
                 world linear velocity (3), body angular velocity (3), absolute height
goal (4)         goal - pelvis position (3, scaled), remaining distance
task (5)         progress fraction, time remaining, floor height ahead,
                 next obstacle rise, distance to the next void
obstacles (12)   three upcoming solids: dx, dy, rise above support, half-width
proprio (57)     joint positions (19), joint velocities (19), previous action (19)

Reward
------
Potential-based progress toward the finish, a small upright/alive shaping term, penalties
for violent actions and torque use, and terminal penalties for falling, touching a hazard
or leaving the course.  Every component is reported separately in ``info`` so a training
run can be diagnosed without guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Sequence

import mujoco
import numpy as np

from .course import CourseScene, build_course
from .humanoid import ActuationMode, HumanoidSpec, default_humanoid
from .levels import Level

ObsMode = Literal["full", "proprio"]

OBSERVATION_NAMES: tuple[str, ...] = (
    "support_height",
    "up_z",
    "body_up_x",
    "body_up_y",
    "body_up_z",
    "linvel_x",
    "linvel_y",
    "linvel_z",
    "angvel_x",
    "angvel_y",
    "angvel_z",
    "pelvis_z",
    "goal_dx",
    "goal_dy",
    "goal_dz",
    "goal_distance",
    "progress_fraction",
    "time_fraction",
    "floor_ahead",
    "next_obstacle_rise",
    "next_void_distance",
    "obs1_dx",
    "obs1_dy",
    "obs1_rise",
    "obs1_halfwidth",
    "obs2_dx",
    "obs2_dy",
    "obs2_rise",
    "obs2_halfwidth",
    "obs3_dx",
    "obs3_dy",
    "obs3_rise",
    "obs3_halfwidth",
)
OBS_JOINT_POS_START = len(OBSERVATION_NAMES)
# joint positions, joint velocities and the previous action follow the fixed task block
OBSERVATION_DIM = OBS_JOINT_POS_START + 19 * 3

ACTION_DIM = 19

TERMINATION_REASONS = (
    "course_complete",
    "fell_over",
    "collapsed",
    "void_fall",
    "hazard_contact",
    "out_of_bounds",
    "timeout",
    "no_progress",
    "none",
)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the shaped reward. All values are per control step unless stated."""

    progress_per_metre: float = 5.0
    forward_velocity_per_second: float = 0.0
    forward_velocity_target: float = 1.2
    gait_per_second: float = 0.0
    alive_per_second: float = 1.0
    upright_per_second: float = 0.5
    height_per_second: float = 1.0
    action_rate_penalty: float = 0.01
    torque_penalty: float = 0.000001
    joint_velocity_penalty: float = 0.00001
    lateral_drift_penalty: float = 0.02
    success_bonus: float = 40.0
    speed_bonus: float = 6.0
    fall_penalty: float = 10.0
    hazard_penalty: float = 12.0
    out_of_bounds_penalty: float = 8.0

    def to_dict(self) -> dict[str, float]:
        return {
            field_name: float(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "RewardConfig":
        known = {key: float(values[key]) for key in cls.__dataclass_fields__ if key in values}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        return cls(**known)

    def scaled(self, factor: float) -> "RewardConfig":
        """Scale every weight (used by bounded evolution over reward magnitudes)."""

        return replace(**{key: value * factor for key, value in self.to_dict().items()})

    @classmethod
    def posture_stage(cls) -> "RewardConfig":
        """Reward used at the start of an annealed run: stand, do not fall.

        Locomotion terms are zero so the policy first discovers a stable stance instead of
        diving at the finish line and learning to fall forward.  ``with`` overrides are
        applied by :func:`parkour_lab.cli.parse_reward_overrides`.
        """

        return cls(
            progress_per_metre=0.0,
            forward_velocity_per_second=0.0,
            gait_per_second=0.0,
            alive_per_second=0.5,
            upright_per_second=3.0,
            height_per_second=3.0,
            action_rate_penalty=0.01,
            torque_penalty=1e-06,
            joint_velocity_penalty=1e-05,
            lateral_drift_penalty=0.02,
            success_bonus=40.0,
            speed_bonus=6.0,
            fall_penalty=10.0,
            hazard_penalty=12.0,
            out_of_bounds_penalty=8.0,
        )

    def blend(self, other: "RewardConfig", alpha: float) -> "RewardConfig":
        """Linear interpolation from ``self`` (alpha=0) to ``other`` (alpha=1)."""

        alpha = float(min(max(alpha, 0.0), 1.0))
        return replace(
            self,
            **{
                key: (1.0 - alpha) * value + alpha * float(getattr(other, key))
                for key, value in self.to_dict().items()
            }
        )


@dataclass(frozen=True, slots=True)
class EnvConfig:
    level_id: str
    seed: int = 0
    control_decimation: int = 5
    physics_timestep: float = 0.002
    action_scale: float = 0.25
    action_mode: Literal["centered", "range"] = "centered"
    action_smoothing: float = 0.6
    actuation: ActuationMode = "position"
    reward: RewardConfig = field(default_factory=RewardConfig)
    randomize_reset: bool = True
    reset_joint_noise_degrees: float = 4.0
    reset_velocity_noise: float = 0.05
    max_episode_seconds: float | None = None
    stagnation_seconds: float = 0.0
    stagnation_progress_rate: float = 0.08
    terminate_on_collapse: bool = True
    collect_diagnostics: bool = True
    record_frames: bool = False
    frame_stride: int = 3

    @property
    def control_dt(self) -> float:
        return self.control_decimation * self.physics_timestep

    def to_dict(self) -> dict[str, Any]:
        data = {
            "level_id": self.level_id,
            "seed": self.seed,
            "control_decimation": self.control_decimation,
            "physics_timestep": self.physics_timestep,
            "action_scale": self.action_scale,
            "action_mode": self.action_mode,
            "action_smoothing": self.action_smoothing,
            "actuation": self.actuation,
            "randomize_reset": self.randomize_reset,
            "reset_joint_noise_degrees": self.reset_joint_noise_degrees,
            "reset_velocity_noise": self.reset_velocity_noise,
            "max_episode_seconds": self.max_episode_seconds,
            "stagnation_seconds": self.stagnation_seconds,
            "stagnation_progress_rate": self.stagnation_progress_rate,
            "terminate_on_collapse": self.terminate_on_collapse,
            "collect_diagnostics": self.collect_diagnostics,
            "record_frames": self.record_frames,
            "frame_stride": self.frame_stride,
            "reward": self.reward.to_dict(),
        }
        return data


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def observation_dim(joint_count: int = ACTION_DIM) -> int:
    """Observation size for a humanoid with ``joint_count`` actuated joints."""

    return OBS_JOINT_POS_START + joint_count * 3


class ParkourEnv:
    """One articulated humanoid on one fixed level."""

    def __init__(
        self,
        level: Level,
        config: EnvConfig | None = None,
        *,
        scene: CourseScene | None = None,
        humanoid: HumanoidSpec | None = None,
        model: mujoco.MjModel | None = None,
    ) -> None:
        self.level = level
        spec = humanoid if humanoid is not None else default_humanoid()
        if config is not None and config.actuation != spec.actuation_mode:
            spec = spec.with_actuation(config.actuation)
        cfg = config if config is not None else EnvConfig(level_id=level.id)
        if cfg.level_id != level.id:
            cfg = replace(cfg, level_id=level.id)
        self.config = cfg
        self.humanoid = spec
        self.scene = scene if scene is not None else build_course(level, spec)
        self.model = (
            model
            if model is not None
            else mujoco.MjModel.from_xml_string(self.scene.xml, {})
        )
        # ``EnvConfig.physics_timestep`` is authoritative.  The MJCF scene carries a default
        # timestep too, and silently keeping it meant every derived quantity (control_dt,
        # episode length, time-based reward terms) disagreed with the actual simulation.
        self.model.opt.timestep = float(cfg.physics_timestep)
        self.data = mujoco.MjData(self.model)

        self._rng = np.random.default_rng(cfg.seed)
        self._setup_indices()
        self._setup_obstacles()
        self._setup_action_mapping()

        self._action = np.zeros(self.action_dim, dtype=np.float64)
        self._previous_action = np.zeros(self.action_dim, dtype=np.float64)
        self._steps = 0
        self._max_steps = int(round(self.time_limit / self.config.control_dt))
        self._initial_goal_distance = 1.0
        self._previous_goal_distance = 1.0
        self._episode_return = 0.0
        self._termination_reason = "none"
        self._last_support = float(level.spawn_support_z)
        self._previous_foot_state: tuple[bool, bool] | None = None
        self._best_goal_distance = float("inf")
        self._progress_rate = 0.0
        self._stagnation_steps = 0
        self._target_pelvis_height = max(
            float(level.spawn_position.z) - float(level.spawn_support_z), 0.1
        )
        self._contact_force = 0.0
        self._air_steps = 0
        self._max_contact_force = 0.0
        self._energy = 0.0
        self._frames: list[dict[str, Any]] = []
        self._episode_seed = int(cfg.seed)
        self._diagnostics: dict[str, float] = {}
        self._setup_buffers()
        self.reset(seed=cfg.seed)

    # ----------------------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------------------
    def _setup_indices(self) -> None:
        model = self.model
        self.joint_qpos_adr = np.array(
            [model.jnt_qposadr[model.joint(name).id] for name in self.humanoid.joint_names],
            dtype=np.int32,
        )
        self.joint_dof_adr = np.array(
            [model.jnt_dofadr[model.joint(name).id] for name in self.humanoid.joint_names],
            dtype=np.int32,
        )
        self.joint_limits = np.array(
            [[joint.lower, joint.upper] for joint in self.humanoid.joints], dtype=np.float64
        )
        self.default_joint_positions = np.deg2rad(
            np.array(self.humanoid.default_joint_positions, dtype=np.float64)
        )
        self.joint_limits_rad = np.deg2rad(self.joint_limits)
        self.actuator_ids = np.array(
            [model.actuator(f"motor_{name}").id for name in self.humanoid.joint_names],
            dtype=np.int32,
        )
        self.robot_body_ids = {
            model.body(name).id for name in self.scene.robot_bodies
        }
        self.robot_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in self.robot_body_ids
        }
        self.foot_geom_ids = {
            model.geom("foot_l_geom").id,
            model.geom("foot_r_geom").id,
        }
        self.left_foot_geom_id = model.geom("foot_l_geom").id
        self.right_foot_geom_id = model.geom("foot_r_geom").id
        self.hazard_geom_ids = {
            model.geom(name).id for name in self.scene.hazard_geoms
        }
        self.pelvis_body_id = model.body("pelvis").id
        self.finish_position = np.array(
            [
                self.level.finish_position.x,
                self.level.finish_position.y,
                self.level.finish_position.z,
            ],
            dtype=np.float64,
        )
        self.spawn_position = np.array(
            [
                self.level.spawn_position.x,
                self.level.spawn_position.y,
                self.level.spawn_position.z,
            ],
            dtype=np.float64,
        )
        self._quat = np.zeros(4, dtype=np.float64)
        self._rotation = np.zeros(9, dtype=np.float64)
        self._force = np.zeros(6, dtype=np.float64)
        self._qpos_slice = slice(0, 3)
        self._quat_slice = slice(3, 7)
        self._joint_qpos_slice = slice(7, 7 + ACTION_DIM)
        self._joint_qvel_slice = slice(6, 6 + ACTION_DIM)

    def _setup_obstacles(self) -> None:
        solids = self.level.solid_obstacles
        self._obstacle_ids = [obstacle.id for obstacle in solids]
        self._obstacle_center = np.array(
            [obstacle.position.as_tuple() for obstacle in solids], dtype=np.float64
        ).reshape(-1, 3)
        self._obstacle_half = np.array(
            [obstacle.size.as_tuple() for obstacle in solids], dtype=np.float64
        ).reshape(-1, 3)
        self._obstacle_hazard = np.array(
            [obstacle.hazard for obstacle in solids], dtype=bool
        )
        self._obstacle_kind = np.array([obstacle.kind for obstacle in solids], dtype=object)
        self._obstacle_tops = self._obstacle_center[:, 2] + self._obstacle_half[:, 2]
        self._probe_offsets = np.arange(0.25, 3.0, 0.25)
        self._foot_site_ids = [
            self.model.site("foot_l_site").id,
            self.model.site("foot_r_site").id,
        ]
        self._foot_site_xpos = np.zeros((2, 3), dtype=np.float64)
        self._moving_indices = []
        self._moving_specs = []
        for move in self.scene.moving_platforms:
            try:
                index = self._obstacle_ids.index(move.obstacle_id)
            except ValueError:  # pragma: no cover - defensive
                continue
            self._moving_indices.append((index, move))
            self._moving_specs.append(move)
        self._mocap_ids = {
            move.body_name: int(self.model.body_mocapid[self.model.body(move.body_name).id])
            for move in self.scene.moving_platforms
        }

    def _setup_action_mapping(self) -> None:
        # ``joint_limits`` is kept in degrees for diagnostics/JSON, while every value that
        # touches the model (qpos, ctrl, targets) is in radians.  Mixing the two here made
        # the effective action range ~57x too large, which is why early PPO runs thrashed.
        lower_rad = self.joint_limits_rad[:, 0]
        upper_rad = self.joint_limits_rad[:, 1]
        if self.config.action_mode == "centered":
            self._action_center = self.default_joint_positions.copy()
            self._action_span = (
                (upper_rad - lower_rad) * 0.5 * self.config.action_scale
            )
        else:
            self._action_center = 0.5 * (upper_rad + lower_rad)
            self._action_span = (upper_rad - lower_rad) * 0.5
        if self.config.actuation == "position":
            self._ctrl_offset = np.zeros(self.action_dim)
            self._ctrl_gain = np.ones(self.action_dim)
        else:
            self._ctrl_offset = np.zeros(self.action_dim)
            self._ctrl_gain = np.ones(self.action_dim)

    def _setup_buffers(self) -> None:
        self.observation = np.zeros(self.observation_dim, dtype=np.float64)
        self._qpos = np.zeros(self.model.nq, dtype=np.float64)
        self._qvel = np.zeros(self.model.nv, dtype=np.float64)
        self._torque = np.zeros(self.action_dim, dtype=np.float64)
        self._ctrl = np.zeros(self.action_dim, dtype=np.float64)

    # ----------------------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return len(self.humanoid.joints)

    @property
    def observation_dim(self) -> int:
        return observation_dim(self.action_dim)

    @property
    def time_limit(self) -> float:
        limit = self.config.max_episode_seconds
        return self.level.time_limit_seconds if limit is None else min(limit, self.level.time_limit_seconds)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def termination_reason(self) -> str:
        return self._termination_reason

    @property
    def action(self) -> np.ndarray:
        return self._action

    @property
    def episode_return(self) -> float:
        return self._episode_return

    @property
    def frames(self) -> list[dict[str, Any]]:
        return self._frames

    @property
    def diagnostics(self) -> dict[str, float]:
        return dict(self._diagnostics)

    # ----------------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = options or {}
        if seed is None:
            self._episode_seed = int(self._rng.integers(0, 2**31 - 1))
        else:
            self._episode_seed = int(seed)
            self._rng = np.random.default_rng(self._episode_seed)
        deterministic = bool(options.get("deterministic", not self.config.randomize_reset))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        yaw = math.radians(self.level.spawn_yaw_degrees)
        quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        self.data.qpos[self._qpos_slice] = self.spawn_position
        self.data.qpos[self._quat_slice] = quat
        joint_targets = self.default_joint_positions.copy()
        if not deterministic and self.config.reset_joint_noise_degrees > 0:
            noise = np.deg2rad(
                self._rng.normal(0.0, self.config.reset_joint_noise_degrees, self.action_dim)
            )
            joint_targets = joint_targets + noise
        self.data.qpos[self._joint_qpos_slice] = np.clip(
            joint_targets, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1]
        )
        if not deterministic and self.config.reset_velocity_noise > 0:
            self.data.qvel[self._joint_qvel_slice] = self._rng.normal(
                0.0, self.config.reset_velocity_noise, self.action_dim
            )
        self.data.ctrl[:] = np.clip(
            self.default_joint_positions, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1]
        )
        self._update_mocap(0.0)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._episode_return = 0.0
        self._termination_reason = "none"
        self._action[:] = 0.0
        self._previous_action[:] = 0.0
        self._ctrl[:] = np.clip(
            self.default_joint_positions, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1]
        )
        self._frames = []
        self._air_steps = 0
        self._max_contact_force = 0.0
        self._energy = 0.0
        self._contact_force = 0.0
        self._diagnostics = {}
        self._previous_foot_state: tuple[bool, bool] | None = None
        self._initial_goal_distance = max(self._goal_distance(), 1e-6)
        self._best_goal_distance = self._initial_goal_distance
        self._progress_rate = 0.0
        self._stagnation_steps = 0
        self._previous_goal_distance = self._initial_goal_distance
        self._last_support = self._support_under_pelvis()
        if math.isfinite(self._last_support):
            self._target_pelvis_height = float(self.data.qpos[2]) - self._last_support
        else:
            self._target_pelvis_height = 0.94
        self._max_steps = max(1, int(round(self.time_limit / self.config.control_dt)))
        observation = self._build_observation()
        info = {
            "level_id": self.level.id,
            "seed": self._episode_seed,
            "step": 0,
            "termination_reason": "none",
            "support_height": self._last_support,
            "time_limit_seconds": self.time_limit,
            "deterministic": deterministic,
        }
        if self.config.record_frames:
            self._capture_frame()
        return observation.copy(), info

    # ----------------------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------------------
    def step(self, action: Sequence[float]) -> StepResult:
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size != self.action_dim:
            raise ValueError(f"expected action of size {self.action_dim}, got {raw.size}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("action contains non-finite values")
        clipped = np.clip(raw, -1.0, 1.0)
        saturation = float(np.mean(np.abs(raw - clipped) > 1e-9))
        self._previous_action = self._action.copy()
        self._action = clipped

        ctrl = self._action_to_ctrl(clipped)
        self._ctrl[:] = ctrl
        self.data.ctrl[self.actuator_ids] = ctrl

        for _ in range(self.config.control_decimation):
            self._update_mocap(float(self.data.time))
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        support = self._support_under_pelvis()
        self._last_support = support
        contact_force, hazard_hit, foot_contact, foot_state = self._scan_contacts()
        self._contact_force = contact_force
        self._max_contact_force = max(self._max_contact_force, contact_force)
        if not foot_contact:
            self._air_steps += 1

        self._track_progress()
        components = self._reward_components(support, contact_force, saturation)
        gait = self._gait_event(foot_state)
        if gait:
            # Credit for stepping is gated on forward motion so that marching in place
            # cannot farm the gait term; a floor of 0.25 keeps the gradient alive while the
            # policy still has to learn to move at all.
            speed = float(self.data.qvel[0])
            target = max(self.config.reward.forward_velocity_target, 1e-6)
            factor = min(max(speed / target, 0.0), 1.0)
            components["gait"] = (
                self.config.reward.gait_per_second * self.config.control_dt * factor
            )
        terminated, truncated, reason = self._check_done(support, hazard_hit)
        reward = float(sum(components.values()))
        if reason != "none":
            reward += self._terminal_reward(reason)
            components["terminal"] = self._terminal_reward(reason)
        self._termination_reason = reason
        self._episode_return += reward
        self._energy += float(np.sum(np.abs(self._torque))) * self.config.control_dt

        observation = self._build_observation()
        info: dict[str, Any] = {
            "step": self._steps,
            "termination_reason": reason,
            "reward_components": components,
            "support_height": support,
            "pelvis_height": float(self.data.qpos[2]),
            "forward_speed": float(self.data.qvel[0]),
            "contact_force": contact_force,
            "foot_contact": foot_contact,
            "action_saturation": saturation,
            "hazard_contact": hazard_hit,
            "progress_fraction": self._progress_fraction(),
            "goal_distance": self._goal_distance(),
            "time_limit_seconds": self.time_limit,
        }
        if self.config.record_frames and (
            self._steps % max(1, self.config.frame_stride) == 0 or terminated or truncated
        ):
            self._capture_frame()
        if terminated or truncated:
            self._diagnostics = self._summarise_episode(support)
            info["episode_metrics"] = self._diagnostics
        return StepResult(observation.copy(), reward, terminated, truncated, info)

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map a normalised action to actuator controls.

        ``action_smoothing`` blends the new target into the previous control value.  A
        first-order filter is a cheap way to stop white-noise exploration from shaking the
        robot apart at the control frequency; at 0 it is the raw target.
        """

        if self.config.actuation == "position":
            targets = self._action_center + self._action_span * action
            targets = np.clip(targets, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1])
        else:
            # muscle mode: normalised torque command
            targets = np.clip(action, -1.0, 1.0)
        smoothing = float(np.clip(self.config.action_smoothing, 0.0, 0.99))
        if smoothing <= 0.0 or self._ctrl.size == 0:
            return targets
        ctrl = (1.0 - smoothing) * self._ctrl + smoothing * targets
        return np.clip(ctrl, self.model.actuator_ctrlrange[self.actuator_ids, 0],
                       self.model.actuator_ctrlrange[self.actuator_ids, 1])

    def _update_mocap(self, time_seconds: float) -> None:
        for index, move in self._moving_indices:
            position = move.position_at(time_seconds)
            mocap_id = self._mocap_ids[move.body_name]
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
            self._obstacle_center[index] = position
            self._obstacle_tops[index] = position[2] + self._obstacle_half[index, 2]

    # ----------------------------------------------------------------------------------
    # Observations
    # ----------------------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        data = self.data
        obs = self.observation
        obs[:] = 0.0
        quat = data.qpos[3:7]
        self._quat[:] = quat
        mujoco.mju_quat2Mat(self._rotation, self._quat)
        rotation = self._rotation.reshape(3, 3)
        up_body = rotation.T @ np.array([0.0, 0.0, 1.0])
        support = self._last_support
        height = float(data.qpos[2] - support) if math.isfinite(support) else 0.0
        linear = data.qvel[0:3]
        angular_body = rotation.T @ data.qvel[3:6]

        obs[0] = _clip(height, 0.0, 2.0)
        obs[1] = _clip(float(up_body[2]), -1.0, 1.0)
        obs[2:5] = np.clip(up_body, -1.0, 1.0)
        obs[5:8] = np.clip(linear, -3.0, 3.0) * 0.5
        obs[8:11] = np.clip(angular_body, -6.0, 6.0) * 0.2
        obs[11] = _clip(float(data.qpos[2]) / 4.0, -1.0, 2.0)

        delta = self.finish_position - data.qpos[0:3]
        obs[12:15] = np.clip(delta, -6.0, 6.0) / 4.0
        obs[15] = _clip(float(np.linalg.norm(delta[:2])) / 12.0, 0.0, 1.0)
        obs[16] = self._progress_fraction()
        obs[17] = _clip(1.0 - self._steps / max(1, self._max_steps), 0.0, 1.0)
        floor_ahead = self._support_height(float(data.qpos[0]) + 0.75, float(data.qpos[1]), 0.15)
        obs[18] = (
            _clip(floor_ahead - float(data.qpos[2]), -1.5, 1.5)
            if math.isfinite(floor_ahead)
            else -1.5
        )
        rise, distance_to_void = self._ahead_features(float(data.qpos[0]))
        obs[19] = _clip(rise, -1.0, 1.5)
        obs[20] = _clip(distance_to_void / 5.0, 0.0, 1.0)

        for slot, index in enumerate(self._upcoming_obstacles(float(data.qpos[0]), 3)):
            base = 21 + slot * 4
            center = self._obstacle_center[index]
            half = self._obstacle_half[index]
            obs[base] = _clip((center[0] - data.qpos[0]) / 5.0, -1.0, 2.0)
            obs[base + 1] = _clip((center[1] - data.qpos[1]) / 2.5, -2.0, 2.0)
            top = center[2] + half[2]
            obs[base + 2] = _clip(top - max(support, 0.0), -1.0, 1.5)
            obs[base + 3] = _clip(half[1] / 2.0, 0.0, 1.0)

        qpos = data.qpos[self.joint_qpos_adr]
        qvel = data.qvel[self.joint_dof_adr]
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        normalized = 2.0 * (qpos - lower) / np.maximum(upper - lower, 1e-6) - 1.0
        start = OBS_JOINT_POS_START
        obs[start : start + self.action_dim] = np.clip(normalized, -1.0, 1.0)
        obs[start + self.action_dim : start + 2 * self.action_dim] = np.clip(
            qvel * 0.1, -5.0, 5.0
        )
        obs[start + 2 * self.action_dim : start + 3 * self.action_dim] = self._action
        return obs

    def _upcoming_obstacles(self, x: float, count: int) -> list[int]:
        candidates = [
            index
            for index in range(len(self._obstacle_ids))
            if not self._obstacle_hazard[index]
            and (self._obstacle_center[index][0] + self._obstacle_half[index][0]) >= x - 0.05
        ]
        candidates.sort(
            key=lambda index: (
                self._obstacle_center[index][0] - self._obstacle_half[index][0],
                index,
            )
        )
        return candidates[:count]

    def _ahead_features(self, x: float) -> tuple[float, float]:
        probes = self._probe_offsets + x
        tops = self._support_heights(probes, float(self.data.qpos[1]), 0.0)
        finite = np.isfinite(tops)
        distance_to_void = (
            float(self._probe_offsets[~finite].min()) if not bool(finite.all()) else 5.0
        )
        if not bool(finite.any()):
            return 0.0, distance_to_void
        reference = self._last_support if math.isfinite(self._last_support) else 0.0
        rise = float(np.max(tops[finite]) - reference)
        return rise, distance_to_void

    # ----------------------------------------------------------------------------------
    # Support / contacts
    # ----------------------------------------------------------------------------------
    def _support_heights(self, xs: np.ndarray, y: float, margin: float = 0.0) -> np.ndarray:
        """Highest solid surface at each x (fixed y); ``-inf`` means a bottomless gap."""

        if self._obstacle_tops.size == 0:
            return np.full(np.shape(xs), -math.inf)
        xs_arr = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        inside_x = np.abs(xs_arr[:, None] - self._obstacle_center[None, :, 0]) <= (
            self._obstacle_half[None, :, 0] + margin
        )
        inside_y = np.abs(y - self._obstacle_center[:, 1]) <= (
            self._obstacle_half[:, 1] + margin
        )
        inside = inside_x & inside_y[None, :]
        tops = np.where(inside, self._obstacle_tops[None, :], -math.inf)
        return tops.max(axis=1)

    def _support_height(self, x: float, y: float, margin: float = 0.0) -> float:
        return float(self._support_heights(np.array([x]), y, margin)[0])

    def _support_under_pelvis(self) -> float:
        """Support surface below the feet, falling back to the pelvis footprint."""

        for slot, site_id in enumerate(self._foot_site_ids):
            self._foot_site_xpos[slot] = self.data.site_xpos[site_id]
        foot_sites = self._foot_site_xpos
        tops = [
            self._support_height(float(foot_sites[i][0]), float(foot_sites[i][1]), 0.06)
            for i in range(foot_sites.shape[0])
        ]
        finite = [top for top in tops if math.isfinite(top)]
        if finite:
            return max(finite)
        return self._support_height(float(self.data.qpos[0]), float(self.data.qpos[1]), 0.12)

    def _scan_contacts(self) -> tuple[float, bool, bool, tuple[bool, bool]]:
        """Contact forces, hazard flag, any-foot flag and the per-foot contact state."""

        total_force = 0.0
        hazard_hit = False
        foot_contact = False
        left = False
        right = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in self.robot_geom_ids and geom2 not in self.robot_geom_ids:
                continue
            if geom1 in self.hazard_geom_ids or geom2 in self.hazard_geom_ids:
                hazard_hit = True
            if geom1 in self.foot_geom_ids or geom2 in self.foot_geom_ids:
                foot_contact = True
                if self.left_foot_geom_id in (geom1, geom2):
                    left = True
                if self.right_foot_geom_id in (geom1, geom2):
                    right = True
            if self.config.collect_diagnostics:
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                total_force += float(np.linalg.norm(force[:3]))
        return total_force, hazard_hit, foot_contact, (left, right)

    # ----------------------------------------------------------------------------------
    # Reward and termination
    # ----------------------------------------------------------------------------------
    def _goal_distance(self) -> float:
        delta = self.finish_position[:2] - self.data.qpos[:2]
        return float(np.linalg.norm(delta))

    def _progress_fraction(self) -> float:
        remaining = self._goal_distance()
        fraction = 1.0 - remaining / max(self._initial_goal_distance, 1e-6)
        return _clip(fraction, 0.0, 1.0)

    def _reward_components(
        self, support: float, contact_force: float, saturation: float
    ) -> dict[str, float]:
        cfg = self.config.reward
        dt = self.config.control_dt
        distance = self._goal_distance()
        progress = self._previous_goal_distance - distance
        self._previous_goal_distance = distance
        up = self._up_z()
        upright = _clip((up - 0.55) / 0.45, 0.0, 1.0)
        qvel = self.data.qvel[self.joint_dof_adr]
        torque = self.data.actuator_force[self.actuator_ids]
        self._torque = np.asarray(torque, dtype=np.float64)
        action_delta = self._action - self._previous_action
        components = {
            "progress": cfg.progress_per_metre * progress,
            "forward_velocity": cfg.forward_velocity_per_second
            * dt
            * float(np.clip(self.data.qvel[0] / max(cfg.forward_velocity_target, 1e-6), 0.0, 1.0)),
            "alive": cfg.alive_per_second * dt,
            "upright": cfg.upright_per_second * dt * upright,
            "height": cfg.height_per_second * dt * self._height_reward(support),
            "action_rate": -cfg.action_rate_penalty * float(np.mean(action_delta**2)),
            "torque": -cfg.torque_penalty * float(np.mean(torque**2)),
            "joint_velocity": -cfg.joint_velocity_penalty * float(np.mean(qvel**2)),
            "lateral_drift": -cfg.lateral_drift_penalty * dt * abs(float(self.data.qvel[1])),
        }
        if saturation > 0:
            components["saturation"] = -0.01 * saturation
        return components

    def _gait_event(self, foot_state: tuple[bool, bool]) -> bool:
        """A "gait event" is a clean exchange of which foot is bearing load.

        Rewarding this directly gives the policy a reason to lift one foot and plant the
        other, which breaks the stand-still local optimum that a pure velocity reward leaves
        behind.  A flight phase (both feet up) is not an event, so hopping is not rewarded.
        """

        previous = self._previous_foot_state
        self._previous_foot_state = foot_state
        if previous is None:
            return False
        left_prev, right_prev = previous
        left_now, right_now = foot_state
        return (
            (right_now and not right_prev and left_prev)  # right heel strike while left stays down
            or (left_now and not left_prev and right_prev)
        )

    def _height_reward(self, support: float) -> float:
        """How close the pelvis is to its standing height above the supporting surface."""

        if not math.isfinite(support):
            return 0.0
        height = float(self.data.qpos[2]) - support
        error = abs(height - self._target_pelvis_height)
        return _clip(1.0 - error / 0.30, 0.0, 1.0)

    def _terminal_reward(self, reason: str) -> float:
        cfg = self.config.reward
        if reason == "course_complete":
            remaining = max(1.0 - self._steps / max(1, self._max_steps), 0.0)
            return cfg.success_bonus + cfg.speed_bonus * remaining
        if reason in ("fell_over", "collapsed", "void_fall"):
            return -cfg.fall_penalty * (0.5 if reason == "collapsed" else 1.0)
        if reason == "hazard_contact":
            return -cfg.hazard_penalty
        if reason == "out_of_bounds":
            return -cfg.out_of_bounds_penalty
        return 0.0

    def _track_progress(self) -> None:
        """Count control steps spent without approaching the finish.

        Locomotion tasks have a stand-still optimum that a dense velocity reward does not
        remove: standing still keeps the posture terms.  Terminating an episode that stops
        approaching the finish makes "do nothing" strictly worse than "keep moving", without
        changing the reward weights themselves.

        The approach rate is smoothed (EMA) so that the pelvis bobbing while standing cannot
        reset the counter, and it is compared against a slow crawl rather than zero.
        """

        rate = (self._previous_goal_distance - self._goal_distance()) / max(
            self.config.control_dt, 1e-9
        )
        self._progress_rate = 0.85 * self._progress_rate + 0.15 * rate
        if self._progress_rate >= self.config.stagnation_progress_rate:
            self._stagnation_steps = 0
        else:
            self._stagnation_steps += 1

    def _check_done(self, support: float, hazard_hit: bool) -> tuple[bool, bool, str]:
        data = self.data
        x, y, z = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
        horizontal = float(np.linalg.norm(self.finish_position[:2] - data.qpos[:2]))
        if (
            horizontal <= self.level.finish_radius
            and abs(z - self.finish_position[2]) < 1.0
        ):
            return True, False, "course_complete"
        if hazard_hit:
            return True, False, "hazard_contact"
        if z < self.level.bounds_min.z + 0.75:
            return True, False, "void_fall"
        if (
            x < self.level.bounds_min.x
            or x > self.level.bounds_max.x
            or y < self.level.bounds_min.y
            or y > self.level.bounds_max.y
        ):
            return True, False, "out_of_bounds"
        if self.config.terminate_on_collapse:
            if self._up_z() < 0.35:
                return True, False, "fell_over"
            if math.isfinite(support) and (z - support) < 0.45:
                return True, False, "collapsed"
        if self._stagnation_steps > 0 and self.config.stagnation_seconds > 0:
            if self._stagnation_steps * self.config.control_dt >= self.config.stagnation_seconds:
                return True, False, "no_progress"
        if self._steps >= self._max_steps:
            return False, True, "timeout"
        return False, False, "none"

    def _up_z(self) -> float:
        quat = self.data.qpos[3:7]
        w, x, y, z = (float(v) for v in quat)
        return _clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)

    # ----------------------------------------------------------------------------------
    # Episode bookkeeping
    # ----------------------------------------------------------------------------------
    def _summarise_episode(self, support: float) -> dict[str, float]:
        duration = self._steps * self.config.control_dt
        return {
            "duration_seconds": duration,
            "completion_fraction": self._progress_fraction(),
            "episode_return": self._episode_return,
            "final_distance": self._goal_distance(),
            "max_contact_force": self._max_contact_force,
            "air_fraction": self._air_steps / max(1, self._steps),
            "energy": self._energy,
            "final_height": float(self.data.qpos[2]),
            "final_support_height": support if math.isfinite(support) else -10.0,
            "success": 1.0 if self._termination_reason == "course_complete" else 0.0,
        }

    def episode_summary(self) -> dict[str, Any]:
        """Serialisable result of the finished (or current) episode."""

        return {
            "level_id": self.level.id,
            "seed": self._episode_seed,
            "success": self._termination_reason == "course_complete",
            "termination_reason": self._termination_reason,
            "completion_fraction": self._progress_fraction(),
            "episode_return": self._episode_return,
            "sim_steps": self._steps,
            "elapsed_seconds": self._steps * self.config.control_dt,
            "metrics": self._diagnostics or self._summarise_episode(self._last_support),
        }

    def _capture_frame(self) -> None:
        """Record world poses of every body for the viewer/replay (no physics state)."""

        bodies: dict[str, list[float]] = {}
        for name in self.scene.robot_bodies:
            body_id = self.model.body(name).id
            position = self.data.xpos[body_id]
            quaternion = self.data.xquat[body_id]
            bodies[name] = [
                round(float(position[0]), 5),
                round(float(position[1]), 5),
                round(float(position[2]), 5),
                round(float(quaternion[0]), 5),
                round(float(quaternion[1]), 5),
                round(float(quaternion[2]), 5),
                round(float(quaternion[3]), 5),
            ]
        movers = {}
        for move in self.scene.moving_platforms:
            mocap_id = self._mocap_ids[move.body_name]
            position = self.data.mocap_pos[mocap_id]
            movers[move.obstacle_id] = [
                round(float(position[0]), 4),
                round(float(position[1]), 4),
                round(float(position[2]), 4),
            ]
        self._frames.append(
            {
                "t": round(float(self._steps * self.config.control_dt), 4),
                "bodies": bodies,
                "movers": movers,
            }
        )

    # ----------------------------------------------------------------------------------
    # State access (replay, GUI, checkpointed evaluation)
    # ----------------------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "time": float(self.data.time),
            "steps": self._steps,
            "previous_action": self._previous_action.tolist(),
            "previous_goal_distance": self._previous_goal_distance,
            "episode_return": self._episode_return,
            "rng_state": self._rng.bit_generator.state,
            "episode_seed": self._episode_seed,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.data.qpos[:] = np.asarray(state["qpos"], dtype=np.float64)
        self.data.qvel[:] = np.asarray(state["qvel"], dtype=np.float64)
        self.data.time = float(state.get("time", 0.0))
        self._steps = int(state.get("steps", 0))
        self._previous_action[:] = np.asarray(
            state.get("previous_action", np.zeros(self.action_dim)), dtype=np.float64
        )
        self._previous_goal_distance = float(
            state.get("previous_goal_distance", self._goal_distance())
        )
        self._episode_return = float(state.get("episode_return", 0.0))
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]
        self._episode_seed = int(state.get("episode_seed", self.config.seed))
        self._last_support = self._support_under_pelvis()
        mujoco.mj_forward(self.model, self.data)

    def body_poses(self) -> dict[str, list[float]]:
        poses: dict[str, list[float]] = {}
        for name in self.scene.robot_bodies:
            body_id = self.model.body(name).id
            position = self.data.xpos[body_id]
            quaternion = self.data.xquat[body_id]
            poses[name] = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ]
        return poses

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Offscreen RGB frame; requires a working OpenGL backend (not needed for training)."""

        import os

        backend = os.environ.get("MUJOCO_GL", "glfw")
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if backend == "glfw" and not display:
            raise RuntimeError(
                "no display for the GLFW renderer; set MUJOCO_GL=egl or osmesa for offscreen "
                "rendering, or use the browser-based replay viewer"
            )
        renderer = mujoco.Renderer(self.model, height, width)
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = self.pelvis_body_id
            camera.distance = 4.0
            camera.azimuth = 140.0
            camera.elevation = -12.0
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                mujoco.MjvOption(),
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                renderer.scene,
            )
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()

    def set_reward(self, reward: RewardConfig) -> None:
        """Replace the reward weights in place (used by annealed reward schedules).

        The reward is a pure function of the state and the weights, so swapping it between
        episodes keeps every other piece of episode state valid.
        """

        self.config = replace(self.config, reward=reward)

    def close(self) -> None:
        """Drop recorded frames; the MuJoCo model/data are owned by this environment."""

        self._frames = []


def _gait_reward_weight(reward: "RewardConfig", dt: float) -> float:
    return reward.gait_per_second * dt


def _clip(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(max(value, low), high))


__all__ = [
    "ACTION_DIM",
    "EnvConfig",
    "OBSERVATION_DIM",
    "OBSERVATION_NAMES",
    "OBS_JOINT_POS_START",
    "ParkourEnv",
    "RewardConfig",
    "StepResult",
    "TERMINATION_REASONS",
    "observation_dim",
]


OBSERVATION_DIM = OBS_JOINT_POS_START + 19 * 3
