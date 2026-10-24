"""Proximal Policy Optimisation with generalised advantage estimation.

A compact, readable implementation for the CPython/NumPy stack used by this project.  It
implements the standard PPO clipped surrogate objective, value-function regression, an
entropy bonus, advantage normalisation, minibatched epochs, gradient-norm clipping and a
learning-rate schedule.  Everything is seeded and every tensor that matters is in the
checkpoint, so a stopped run can be resumed.

The rollout math is backend independent: :func:`compute_gae` and :func:`ppo_update` take
arrays from any environment adapter.  The training loop lives in
:mod:`parkour_lab.training.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .policies import ActorCritic


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters for PPO; all fields are checkpointed in the run configuration."""

    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.003
    value_coefficient: float = 0.5
    value_clip: float = 10.0
    epochs: int = 6
    minibatch_size: int = 512
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    reward_scale: float = 1.0
    target_kl: float | None = 0.04
    learning_rate_final_fraction: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PPOConfig":
        known = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown PPO settings: {sorted(unknown)}")
        return cls(**known)


@dataclass(slots=True)
class Rollout:
    """One on-policy batch of transitions."""

    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray = field(default_factory=lambda: np.zeros(0))
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(self.observations.shape[0])


@dataclass(slots=True)
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
    bootstrap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation for a single environment stream.

    ``dones[t]`` marks that the transition ended an episode, so the recursion is cut.
    ``last_value`` bootstraps the value of the state after the final step.

    ``bootstrap[t]`` is used instead of ``gamma * 0`` for a *truncated* episode: the episode
    stopped because of a time limit, so the state it reached still has value.  ``dones`` must
    still be 1 there, which cuts the advantage recursion but keeps the target
    ``r_t + gamma * V(s_final)``.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    steps = rewards.shape[0]
    if bootstrap is None:
        bootstrap = np.zeros(steps, dtype=np.float64)
    else:
        bootstrap = np.asarray(bootstrap, dtype=np.float64)
    advantages = np.zeros(steps, dtype=np.float64)
    last_advantage = 0.0
    for step in range(steps - 1, -1, -1):
        next_value = last_value if step == steps - 1 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = (
            rewards[step] + gamma * next_value * not_done + bootstrap[step] - values[step]
        )
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    batch: Rollout,
    config: PPOConfig,
    rng: np.random.Generator,
    *,
    learning_rate: float | None = None,
) -> UpdateStats:
    """Run ``config.epochs`` minibatched PPO epochs over ``batch``."""

    observations = np.asarray(batch.observations, dtype=np.float64)
    actions = np.asarray(batch.actions, dtype=np.float64)
    old_log_probs = np.asarray(batch.log_probs, dtype=np.float64)
    advantages = np.asarray(batch.advantages, dtype=np.float64)
    returns = np.asarray(batch.returns, dtype=np.float64)

    if config.normalize_advantage and advantages.size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    sample_count = observations.shape[0]
    lr = config.learning_rate if learning_rate is None else float(learning_rate)
    if sample_count == 0:
        return UpdateStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lr)
    minibatch = max(1, min(config.minibatch_size, sample_count))
    lr_scale = lr / config.learning_rate if config.learning_rate > 0 else 1.0
    clip_low = 1.0 - config.clip_ratio
    clip_high = 1.0 + config.clip_ratio
    action_dim = actions.shape[1] if actions.ndim == 2 else model.action_dim

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []

    for _epoch in range(config.epochs):
        order = rng.permutation(sample_count)
        stop_early = False
        for start in range(0, sample_count, minibatch):
            index = order[start : start + minibatch]
            obs = observations[index]
            act = actions[index]
            old_log = old_log_probs[index]
            adv = advantages[index]
            ret = returns[index]
            adv_var = float(np.var(ret))

            log_prob, values, mean = model.evaluate_actions(obs, act)
            ratio = np.exp(np.clip(log_prob - old_log, -20.0, 20.0))
            unclipped = ratio * adv
            clipped = np.clip(ratio, clip_low, clip_high) * adv
            policy_loss = -float(np.mean(np.minimum(unclipped, clipped)))

            # Which branch of min() won determines whether the ratio carries gradient.
            active = ((adv > 0) & (ratio < clip_high)) | ((adv < 0) & (ratio > clip_low))
            d_log_prob = np.where(active, -adv * ratio, 0.0)

            value_delta = values - ret
            value_loss = 0.5 * float(np.mean(value_delta**2))
            d_value = config.value_coefficient * value_delta / values.shape[0]

            actor_grads = model.actor_gradients(obs, act, d_log_prob)
            critic_grads = model.critic_gradients(obs, d_value)
            log_std_grad = model.log_std_gradient(
                act, mean, d_log_prob, entropy_coefficient=config.entropy_coefficient
            )
            model.apply_gradients(
                actor_grads,
                critic_grads,
                log_std_grad,
                max_grad_norm=config.max_grad_norm,
                learning_rate_scale=lr_scale,
            )

            entropy = float(
                np.sum(model.log_std) + 0.5 * action_dim * np.log(2 * np.pi * np.e)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                approx_kl = float(np.mean(old_log - log_prob))
            clip_fraction = float(np.mean(np.abs(ratio - 1.0) > config.clip_ratio))
            del adv_var
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            kls.append(approx_kl)
            clip_fractions.append(clip_fraction)
            if (
                config.target_kl is not None
                and np.isfinite(approx_kl)
                and approx_kl > 1.5 * config.target_kl
            ):
                stop_early = True
                break
        if stop_early:
            break

    prediction = np.asarray(batch.values, dtype=np.float64)
    target = np.asarray(batch.returns, dtype=np.float64)
    variance = float(np.var(target))
    explained = (
        float(1.0 - np.var(target - prediction) / variance) if variance > 1e-12 else 0.0
    )
    return UpdateStats(
        policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
        value_loss=float(np.mean(value_losses)) if value_losses else 0.0,
        entropy=float(np.mean(entropies)) if entropies else 0.0,
        approx_kl=float(np.mean(kls)) if kls else 0.0,
        clip_fraction=float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        explained_variance=explained,
        learning_rate=lr,
    )


def linear_schedule(
    start: float, end_fraction: float, fraction: float
) -> float:
    """Linear interpolation of a hyperparameter from ``start`` to ``start*fraction``."""

    fraction = float(min(max(fraction, 0.0), 1.0))
    return float(start * (1.0 - (1.0 - end_fraction) * fraction))


__all__ = ["PPOConfig", "Rollout", "UpdateStats", "compute_gae", "linear_schedule", "ppo_update"]
