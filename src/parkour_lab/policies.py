"""NumPy actor-critic policy, observation normalisation and bounded checkpoints.

The project deliberately keeps its learning code small and dependency-light: a two-layer
tanh MLP with a Gaussian action head, trained by the PPO implementation in
:mod:`parkour_lab.ppo`.  No deep-learning framework is required, so the same code runs on
CPU-only machines and inside multiprocessing workers.

A checkpoint is a single ``.npz`` file containing every parameter, the optimiser moments,
the observation-normalisation statistics and the training counter.  Resuming restores all
of it, so a resumed run is the same computation as an uninterrupted one (up to floating
point and RNG stream bookkeeping).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PARAM_FORMAT_VERSION = 1


def orthogonal_init(
    rng: np.random.Generator, shape: tuple[int, ...], gain: float = 1.0
) -> np.ndarray:
    """Orthogonal initialisation for a (fan_in, fan_out) weight matrix."""

    rows, cols = shape
    if rows < cols:
        matrix = rng.standard_normal((cols, rows))
        q, r = np.linalg.qr(matrix)
        q = q * np.sign(np.diag(r))
        result = q.T
    else:
        matrix = rng.standard_normal((rows, cols))
        q, r = np.linalg.qr(matrix)
        q = q * np.sign(np.diag(r))
        result = q
    return (gain * result).astype(np.float64)


class Adam:
    """Minimal Adam optimiser with gradient clipping, used by :class:`MLP`."""

    def __init__(
        self,
        shapes: Iterable[tuple[int, ...]],
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-5,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2 = betas
        self.epsilon = float(epsilon)
        self.m = [np.zeros(shape, dtype=np.float64) for shape in shapes]
        self.v = [np.zeros(shape, dtype=np.float64) for shape in shapes]
        self.t = 0

    def step(
        self,
        params: list[np.ndarray],
        grads: list[np.ndarray],
        *,
        max_norm: float = 0.5,
        scale: float = 1.0,
    ) -> None:
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if norm > max_norm and norm > 0:
            factor = max_norm / norm
            grads = [g * factor for g in grads]
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            param -= scale * self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def state_dict(self) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {"t": np.array([self.t])}
        for index, value in enumerate(self.m):
            data[f"adam_m_{index}"] = value
        for index, value in enumerate(self.v):
            data[f"adam_v_{index}"] = value
        return data

    def load_state_dict(self, data: dict[str, np.ndarray]) -> None:
        if "t" in data:
            self.t = int(data["t"][0])
        for index in range(len(self.m)):
            key_m = f"adam_m_{index}"
            key_v = f"adam_v_{index}"
            if key_m in data:
                self.m[index] = np.array(data[key_m], dtype=np.float64)
            if key_v in data:
                self.v[index] = np.array(data[key_v], dtype=np.float64)


class MLP:
    """Two-hidden-layer tanh MLP with manual backpropagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        output_gain: float = 0.01,
        hidden_layers: int = 2,
    ) -> None:
        sizes = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        self.weights = [
            orthogonal_init(rng, (sizes[i], sizes[i + 1]), gain=np.sqrt(2.0))
            for i in range(len(sizes) - 2)
        ]
        self.biases = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 2)]
        self.weights.append(orthogonal_init(rng, (sizes[-2], sizes[-1]), gain=output_gain))
        self.biases.append(np.zeros(sizes[-1]))
        self.sizes = sizes
        self.activations: list[np.ndarray] = []

    # ----------------------------------------------------------------------------------
    def parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = []
        for weight, bias in zip(self.weights, self.biases):
            params.append(weight)
            params.append(bias)
        return params

    def forward(self, observations: np.ndarray) -> np.ndarray:
        activation = np.asarray(observations, dtype=np.float64)
        self.activations = [activation]
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activation = activation @ weight + bias
            if index != last:
                activation = np.tanh(activation)
                self.activations.append(activation)
        return activation

    def backward(self, grad_output: np.ndarray) -> list[np.ndarray]:
        grads: list[np.ndarray] = []
        batch = grad_output.shape[0]
        last = len(self.weights) - 1
        grad = grad_output
        for index in range(last, -1, -1):
            activation = self.activations[index]
            weight = self.weights[index]
            grad_w = activation.T @ grad
            grad_b = grad.sum(axis=0)
            grads.insert(0, grad_b)
            grads.insert(0, grad_w)
            if index > 0:
                grad = (grad @ weight.T) * (1.0 - activation * activation)
        del batch
        return grads


@dataclass
class RunningNorm:
    """Running mean/variance for observation normalisation."""

    size: int
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: float = field(default=1e-4)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.var = np.ones(self.size, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.size == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, observations: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (np.asarray(observations, dtype=np.float64) - self.mean) / np.sqrt(
            self.var + 1e-8
        )
        return np.clip(normalized, -clip, clip)

    def state_dict(self, prefix: str = "obs_norm_") -> dict[str, np.ndarray]:
        return {
            f"{prefix}mean": self.mean,
            f"{prefix}var": self.var,
            f"{prefix}count": np.array([self.count]),
        }

    def load_state_dict(self, data: dict[str, np.ndarray], prefix: str = "obs_norm_") -> None:
        if f"{prefix}mean" in data:
            self.mean = np.array(data[f"{prefix}mean"], dtype=np.float64)
            self.var = np.array(data[f"{prefix}var"], dtype=np.float64)
            self.count = float(data[f"{prefix}count"][0])


class ActorCritic:
    """Gaussian actor with a separate value critic and observation normalisation."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 128,
        seed: int = 0,
        learning_rate: float = 3e-4,
        initial_log_std: float = -0.5,
        normalize_observations: bool = True,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.normalize_observations = bool(normalize_observations)
        rng = np.random.default_rng(seed)
        self.actor = MLP(self.observation_dim, hidden_dim, self.action_dim, rng)
        self.critic = MLP(self.observation_dim, hidden_dim, 1, rng, output_gain=1.0)
        self.log_std = np.full(self.action_dim, float(initial_log_std))
        self.obs_norm = RunningNorm(self.observation_dim)
        self.actor_adam = Adam([p.shape for p in self.actor.parameters()], learning_rate)
        self.critic_adam = Adam([p.shape for p in self.critic.parameters()], learning_rate)
        self.log_std_adam = Adam([self.log_std.shape], learning_rate)

    # ----------------------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------------------
    def normalize(self, observations: np.ndarray) -> np.ndarray:
        if not self.normalize_observations:
            return np.asarray(observations, dtype=np.float64)
        return self.obs_norm.normalize(observations)

    def value(self, observations: np.ndarray) -> np.ndarray:
        normalized = self.normalize(np.atleast_2d(observations))
        return self.critic.forward(normalized).ravel()

    def action_mean(self, observations: np.ndarray) -> np.ndarray:
        normalized = self.normalize(np.atleast_2d(observations))
        return self.actor.forward(normalized)

    def sample(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw actions for a rollout, clipped to the range the environment executes.

        The environment clips every action to ``[-1, 1]`` (the actuator mapping assumes it),
        so an unclipped sample would mean PPO scores and updates an action that was never
        applied.  Clipping here and evaluating the log-probability *of the clipped action*
        at least keeps (observation, action, log-prob) describing the executed transition.
        The density of a clipped Gaussian is not renormalised over ``[-1, 1]``; that is a
        known approximation, and it is much smaller than acting out of range silently.
        """

        mean = self.action_mean(observations)
        std = np.exp(self.log_std)
        noise = rng.standard_normal(mean.shape)
        actions = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = -0.5 * np.sum(
            ((actions - mean) / std) ** 2 + 2.0 * self.log_std + np.log(2 * np.pi), axis=-1
        )
        values = self.critic.forward(self.normalize(observations)).ravel()
        return actions, log_prob, values

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Inference boundary used by evaluation, replay and the GUI."""

        obs = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        mean = self.action_mean(obs)[0]
        if deterministic or rng is None:
            return np.clip(mean, -1.0, 1.0)
        std = np.exp(self.log_std)
        return np.clip(mean + std * rng.standard_normal(self.action_dim), -1.0, 1.0)

    # ----------------------------------------------------------------------------------
    # Training hooks (used by PPO)
    # ----------------------------------------------------------------------------------
    def evaluate_actions(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = self.normalize(observations)
        mean = self.actor.forward(normalized)
        std = np.exp(self.log_std)
        log_prob = -0.5 * np.sum(
            ((actions - mean) / std) ** 2 + 2.0 * self.log_std + np.log(2 * np.pi), axis=-1
        )
        values = self.critic.forward(normalized).ravel()
        return log_prob, values, mean

    def actor_gradients(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        d_log_prob: np.ndarray,
        d_entropy: float = 0.0,
    ) -> list[np.ndarray]:
        normalized = self.normalize(observations)
        mean = self.actor.forward(normalized)
        std = np.exp(self.log_std)
        d_mean = d_log_prob[:, None] * (actions - mean) / (std**2)
        grads = self.actor.backward(d_mean)
        if d_entropy:
            # d entropy / d mean is zero for a fixed-std Gaussian; only log_std matters.
            pass
        return grads

    def critic_gradients(
        self, observations: np.ndarray, d_value: np.ndarray
    ) -> list[np.ndarray]:
        normalized = self.normalize(observations)
        self.critic.forward(normalized)
        return self.critic.backward(d_value[:, None])

    def log_std_gradient(
        self,
        actions: np.ndarray,
        mean: np.ndarray,
        d_log_prob: np.ndarray,
        *,
        entropy_coefficient: float = 0.0,
    ) -> np.ndarray:
        """Gradient of the PPO loss with respect to ``log_std``.

        ``d_log_prob`` is d(loss)/d(log_prob) per sample.  The Gaussian entropy is
        ``sum(log_std) + const``, and the loss maximises entropy, so the entropy term
        contributes ``-entropy_coefficient`` per action dimension.
        """

        std = np.exp(self.log_std)
        z = (actions - mean) / std
        grad = np.sum(d_log_prob[:, None] * (z * z - 1.0), axis=0)
        grad -= entropy_coefficient * np.ones_like(self.log_std)
        return grad

    def apply_gradients(
        self,
        actor_grads: list[np.ndarray],
        critic_grads: list[np.ndarray],
        log_std_grad: np.ndarray | None,
        *,
        max_grad_norm: float = 0.5,
        learning_rate_scale: float = 1.0,
    ) -> None:
        self.actor_adam.step(
            self.actor.parameters(),
            actor_grads,
            max_norm=max_grad_norm,
            scale=learning_rate_scale,
        )
        self.critic_adam.step(
            self.critic.parameters(),
            critic_grads,
            max_norm=max_grad_norm,
            scale=learning_rate_scale,
        )
        if log_std_grad is not None:
            self.log_std_adam.step(
                [self.log_std], [log_std_grad], max_norm=max_grad_norm, scale=learning_rate_scale
            )
            np.clip(self.log_std, -4.0, 1.0, out=self.log_std)

    # ----------------------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------------------
    def state_dict(self) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {
            "format_version": np.array([PARAM_FORMAT_VERSION]),
            "observation_dim": np.array([self.observation_dim]),
            "action_dim": np.array([self.action_dim]),
            "hidden_dim": np.array([self.hidden_dim]),
            "log_std": self.log_std,
            "normalize_observations": np.array([1.0 if self.normalize_observations else 0.0]),
        }
        for index, param in enumerate(self.actor.parameters()):
            data[f"actor_{index}"] = param
        for index, param in enumerate(self.critic.parameters()):
            data[f"critic_{index}"] = param
        data.update(self.obs_norm.state_dict())
        data.update(self.actor_adam.state_dict())
        data.update({f"critic_{key}": value for key, value in self.critic_adam.state_dict().items()})
        data.update({f"logstd_{key}": value for key, value in self.log_std_adam.state_dict().items()})
        return data

    def load_state_dict(self, data: dict[str, np.ndarray]) -> None:
        self.log_std = np.array(data["log_std"], dtype=np.float64)
        for index, param in enumerate(self.actor.parameters()):
            key = f"actor_{index}"
            if key in data:
                param[...] = data[key]
        for index, param in enumerate(self.critic.parameters()):
            key = f"critic_{index}"
            if key in data:
                param[...] = data[key]
        self.obs_norm.load_state_dict(data)
        self.actor_adam.load_state_dict(data)
        critic_data = {
            key[len("critic_") :]: value
            for key, value in data.items()
            if key.startswith("critic_adam_") or key == "critic_t"
        }
        if critic_data:
            self.critic_adam.load_state_dict(critic_data)
        logstd_data = {
            key[len("logstd_") :]: value
            for key, value in data.items()
            if key.startswith("logstd_")
        }
        if logstd_data:
            self.log_std_adam.load_state_dict(logstd_data)

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        payload = dict(self.state_dict())
        if extra:
            payload["extra_json"] = np.frombuffer(
                json.dumps(extra).encode("utf-8"), dtype=np.uint8
            )
        np.savez_compressed(buffer, **payload)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(buffer.getvalue())
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> tuple["ActorCritic", dict[str, Any]]:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        observation_dim = int(arrays["observation_dim"][0])
        action_dim = int(arrays["action_dim"][0])
        hidden_dim = int(arrays["hidden_dim"][0])
        normalize = bool(arrays["normalize_observations"][0])
        model = cls(
            observation_dim,
            action_dim,
            hidden_dim=hidden_dim,
            seed=seed,
            normalize_observations=normalize,
        )
        model.load_state_dict(arrays)
        extra: dict[str, Any] = {}
        if "extra_json" in arrays:
            extra = json.loads(bytes(arrays["extra_json"]).decode("utf-8"))
        return model, extra


__all__ = ["ActorCritic", "Adam", "MLP", "PARAM_FORMAT_VERSION", "RunningNorm", "orthogonal_init"]
