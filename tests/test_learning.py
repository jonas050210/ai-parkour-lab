"""Policy, PPO and vector-environment tests.

The PPO tests use finite differences on the actual NumPy implementation rather than
trusting the maths by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from parkour_lab.env import ACTION_DIM, OBSERVATION_DIM, EnvConfig
from parkour_lab.levels import LevelCatalog
from parkour_lab.policies import MLP, ActorCritic, Adam, RunningNorm, orthogonal_init
from parkour_lab.ppo import PPOConfig, Rollout, compute_gae, linear_schedule, ppo_update
from parkour_lab.training.curriculum import (
    AdaptiveLevelSampler,
    CurriculumConfig,
    SingleLevelSampler,
    UniformLevelSampler,
    make_sampler,
    summarise_performance,
)
from parkour_lab.training.vector import SerialVecEnv, _EnvPool, make_vec_env


@pytest.fixture(scope="module")
def catalog() -> LevelCatalog:
    return LevelCatalog.load()


def _rollout(model: ActorCritic, count: int, seed: int = 0) -> Rollout:
    rng = np.random.default_rng(seed)
    observations = rng.normal(0, 1, (count, model.observation_dim))
    actions = rng.uniform(-0.7, 0.7, (count, model.action_dim))
    log_probs, values, _ = model.evaluate_actions(observations, actions)
    rewards = rng.normal(0, 0.3, count)
    dones = np.zeros(count)
    return Rollout(
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
    )


# ---------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------
def test_orthogonal_init_shapes_and_orthogonality() -> None:
    rng = np.random.default_rng(0)
    matrix = orthogonal_init(rng, (64, 16))
    assert matrix.shape == (64, 16)
    gram = matrix.T @ matrix
    np.testing.assert_allclose(gram, np.eye(16), atol=1e-8)
    tall = orthogonal_init(rng, (8, 32))
    np.testing.assert_allclose(tall @ tall.T, np.eye(8), atol=1e-8)


def test_mlp_backprop_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    net = MLP(4, 6, 2, rng, output_gain=1.0)
    x = rng.normal(0, 1, (3, 4))
    target = rng.normal(0, 1, (3, 2))
    loss = lambda: float(np.mean((net.forward(x) - target) ** 2))  # noqa: E731
    net.forward(x)
    grad_out = 2.0 * (net.forward(x) - target) / target.size
    grads = net.backward(grad_out)
    epsilon = 1e-6
    for index, param in enumerate(net.parameters()):
        assert grads[index].shape == param.shape
        for position in (0, min(3, param.size - 1)):
            cell = np.unravel_index(position, param.shape)
            original = param[cell]
            param[cell] = original + epsilon
            plus = loss()
            param[cell] = original - epsilon
            minus = loss()
            param[cell] = original
            numeric = (plus - minus) / (2 * epsilon)
            assert grads[index].reshape(-1)[position] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_adam_improves_a_quadratic() -> None:
    parameter = np.array([[3.0, -2.0]])
    optimiser = Adam([parameter.shape], learning_rate=0.1)
    for _ in range(200):
        grad = 2.0 * parameter
        optimiser.step([parameter], [grad])
    assert abs(float(parameter[0, 0])) < 1e-2


def test_running_norm_statistics() -> None:
    norm = RunningNorm(3)
    data = np.array([[0.0, 10.0, -2.0], [2.0, 14.0, 2.0], [4.0, 18.0, 6.0]])
    norm.update(data)
    np.testing.assert_allclose(norm.mean, data.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(norm.var, data.var(axis=0), rtol=5e-3)
    normalised = norm.normalize(data)
    assert np.all(np.abs(normalised) < 5)


# ---------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------
def test_policy_sampling_shapes_and_log_prob_consistency() -> None:
    model = ActorCritic(OBSERVATION_DIM, ACTION_DIM, hidden_dim=32, seed=3)
    observations = np.random.default_rng(0).normal(0, 1, OBSERVATION_DIM)
    rng = np.random.default_rng(4)
    actions, log_prob, value = model.sample(observations, rng)
    assert actions.shape == (1, ACTION_DIM)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)
    recomputed, values, mean = model.evaluate_actions(observations[None, :], actions)
    assert float(recomputed[0]) == pytest.approx(float(log_prob[0]), rel=1e-9)
    assert float(values[0]) == pytest.approx(float(value[0]), rel=1e-9)
    assert mean.shape == actions.shape


def test_policy_actor_gradients_match_finite_differences() -> None:
    model = ActorCritic(5, 2, hidden_dim=8, seed=1)
    rng = np.random.default_rng(0)
    observations = rng.normal(0, 1, (4, 5))
    actions = rng.uniform(-0.5, 0.5, (4, 2))
    weights = rng.normal(0, 1, 4)

    log_prob, _, _ = model.evaluate_actions(observations, actions)
    d_log_prob = -weights
    grads = model.actor_gradients(observations, actions, d_log_prob)

    def loss() -> float:
        log_prob, _, _ = model.evaluate_actions(observations, actions)
        return float(-np.sum(log_prob * weights))

    epsilon = 1e-6
    for index, parameter in enumerate(model.actor.parameters()):
        for position in range(min(3, parameter.size)):
            cell = np.unravel_index(position, parameter.shape)
            original = parameter[cell]
            parameter[cell] = original + epsilon
            plus = loss()
            parameter[cell] = original - epsilon
            minus = loss()
            parameter[cell] = original
            numeric = (plus - minus) / (2 * epsilon)
            assert grads[index].reshape(-1)[position] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_log_std_gradient_matches_finite_differences() -> None:
    model = ActorCritic(5, 2, hidden_dim=8, seed=2)
    rng = np.random.default_rng(2)
    observations = rng.normal(0, 1, (3, 5))
    actions = rng.uniform(-0.4, 0.4, (3, 2))
    weights = rng.normal(0, 1, 3)
    _, _, mean = model.evaluate_actions(observations, actions)
    grad = model.log_std_gradient(actions, mean, -weights, entropy_coefficient=0.01)

    def loss() -> float:
        log_prob, _, _ = model.evaluate_actions(observations, actions)
        entropy = float(np.sum(model.log_std))
        return float(-np.sum(log_prob * weights) - 0.01 * entropy)

    epsilon = 1e-6
    for index in range(model.action_dim):
        original = model.log_std[index]
        model.log_std[index] = original + epsilon
        plus = loss()
        model.log_std[index] = original - epsilon
        minus = loss()
        model.log_std[index] = original
        numeric = (plus - minus) / (2 * epsilon)
        assert grad[index] == pytest.approx(numeric, rel=1e-4, abs=1e-5)


def test_checkpoint_round_trip_preserves_behaviour(tmp_path) -> None:
    model = ActorCritic(OBSERVATION_DIM, ACTION_DIM, hidden_dim=32, seed=5, initial_log_std=-0.7)
    rng = np.random.default_rng(0)
    model.obs_norm.update(rng.normal(0, 1, (64, OBSERVATION_DIM)))
    model.log_std[:] = -0.42
    path = tmp_path / "policy.npz"
    model.save(path, extra={"note": "round-trip"})
    restored, extra = ActorCritic.load(path)
    observations = rng.normal(0, 1, (2, OBSERVATION_DIM))
    np.testing.assert_allclose(model.action_mean(observations), restored.action_mean(observations))
    np.testing.assert_allclose(model.value(observations), restored.value(observations))
    np.testing.assert_allclose(restored.log_std, model.log_std)
    assert extra["note"] == "round-trip"
    assert restored.hidden_dim == 32


# ---------------------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------------------
def test_gae_matches_manual_recursion() -> None:
    rewards = np.array([1.0, 0.5, -1.0, 0.0])
    values = np.array([2.0, 1.5, 1.0, 0.5])
    dones = np.array([0.0, 0.0, 1.0, 0.0])
    advantages, returns = compute_gae(
        rewards, values, dones, gamma=0.9, gae_lambda=0.8, last_value=0.2
    )
    expected = np.zeros(4)
    running = 0.0
    for step in range(3, -1, -1):
        next_value = 0.2 if step == 3 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = rewards[step] + 0.9 * next_value * not_done - values[step]
        running = delta + 0.9 * 0.8 * not_done * running
        expected[step] = running
    np.testing.assert_allclose(advantages, expected, atol=1e-12)
    np.testing.assert_allclose(returns, expected + values, atol=1e-12)


def test_ppo_update_reduces_value_error_and_keeps_actions_in_distribution() -> None:
    model = ActorCritic(OBSERVATION_DIM, ACTION_DIM, hidden_dim=48, seed=0)
    batch = _rollout(model, 96, seed=1)
    batch.advantages, batch.returns = compute_gae(
        batch.rewards, batch.values, batch.dones, gamma=0.99, gae_lambda=0.95
    )
    config = PPOConfig(epochs=3, minibatch_size=32, learning_rate=3e-4)
    before = float(np.mean((model.value(batch.observations) - batch.returns) ** 2))
    stats = ppo_update(model, batch, config, np.random.default_rng(0))
    after = float(np.mean((model.value(batch.observations) - batch.returns) ** 2))
    assert after <= before
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    assert np.isfinite(stats.approx_kl)
    assert stats.entropy > 0


def test_ppo_update_rejects_empty_batch() -> None:
    model = ActorCritic(OBSERVATION_DIM, ACTION_DIM, hidden_dim=16, seed=0)
    batch = Rollout(
        observations=np.zeros((0, OBSERVATION_DIM)),
        actions=np.zeros((0, ACTION_DIM)),
        log_probs=np.zeros(0),
        values=np.zeros(0),
        rewards=np.zeros(0),
        dones=np.zeros(0),
        advantages=np.zeros(0),
        returns=np.zeros(0),
    )
    stats = ppo_update(model, batch, PPOConfig(), np.random.default_rng(0))
    assert stats.policy_loss == 0.0


def test_linear_schedule_endpoints() -> None:
    assert linear_schedule(1.0, 0.25, 0.0) == pytest.approx(1.0)
    assert linear_schedule(1.0, 0.25, 1.0) == pytest.approx(0.25)
    assert linear_schedule(1.0, 0.25, 2.0) == pytest.approx(0.25)


# ---------------------------------------------------------------------------------------
# Vector environments
# ---------------------------------------------------------------------------------------
def test_vector_env_step_shapes(catalog: LevelCatalog) -> None:
    levels = ["level-01-basics", "level-02-step-up"]
    vec = make_vec_env(
        catalog,
        levels,
        env_config=EnvConfig(level_id=levels[0]),
        seeds=[1, 2],
        workers=1,
    )
    observations = vec.reset()
    assert observations.shape == (2, OBSERVATION_DIM)
    actions = np.zeros((2, ACTION_DIM))
    result = vec.step(actions)
    assert len(result) == 5
    observations, rewards, terminated, truncated, infos = result
    assert observations.shape == (2, OBSERVATION_DIM)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert len(infos) == 2
    vec.close()


def test_env_pool_gives_each_slot_its_own_episode(catalog: LevelCatalog) -> None:
    """Regression: two slots on the same level must not share one MuJoCo world."""

    pool = _EnvPool(
        catalog,
        EnvConfig(level_id="level-01-basics", randomize_reset=False),
        ["level-01-basics", "level-01-basics"],
        [7, 1007],
    )
    pool.reset()
    seen: dict[int, list[int]] = {0: [], 1: []}
    lengths = {0: 0, 1: 0}
    rng = np.random.default_rng(0)
    for _ in range(400):
        _, _, terminated, truncated, infos = pool.step(rng.uniform(-1, 1, (2, ACTION_DIM)))
        for slot, info in enumerate(infos):
            lengths[slot] += 1
            episode = info.get("episode")
            if episode:
                seen[slot].append(int(episode["seed"]))
                assert episode["sim_steps"] > 1
                assert lengths[slot] > 1
                lengths[slot] = 0
    pool.close()
    # both slots must produce independent episodes with different deterministic seeds
    assert seen[0] and seen[1]
    assert set(seen[0]).isdisjoint(set(seen[1]))


def test_serial_and_subprocess_workers_agree(catalog: LevelCatalog) -> None:
    levels = ["level-01-basics", "level-01-basics"]
    config = EnvConfig(level_id=levels[0], randomize_reset=False, action_smoothing=0.5)
    serial = SerialVecEnv(catalog, levels, env_config=config, seeds=[11, 23])
    parallel = make_vec_env(
        catalog, levels, env_config=config, seeds=[11, 23], workers=2
    )
    try:
        obs_a = serial.reset()
        obs_b = parallel.reset()
        np.testing.assert_allclose(obs_a, obs_b, atol=1e-12)
        actions = np.full((2, ACTION_DIM), 0.05)
        for _ in range(5):
            out_a = serial.step(actions)
            out_b = parallel.step(actions)
            np.testing.assert_allclose(out_a[0], out_b[0], atol=1e-9)
            np.testing.assert_allclose(out_a[1], out_b[1], atol=1e-9)
    finally:
        serial.close()
        parallel.close()


def test_vector_env_can_switch_levels(catalog: LevelCatalog) -> None:
    vec = make_vec_env(
        catalog,
        ["level-01-basics", "level-02-step-up"],
        env_config=EnvConfig(level_id="level-01-basics"),
        seeds=[5, 6],
        workers=1,
    )
    vec.reset()
    vec.set_levels(["level-03-small-hops", "level-04-balance-beam"])
    for _ in range(3):
        vec.step(np.zeros((2, ACTION_DIM)))
    assert vec.levels == ["level-03-small-hops", "level-04-balance-beam"]
    vec.close()


# ---------------------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------------------
def test_curriculum_samplers(catalog: LevelCatalog) -> None:
    levels = list(catalog.train_levels[:4])
    ids = [level.id for level in levels]
    single = SingleLevelSampler(levels[0].id)
    assert single.sample(2) == [ids[0], ids[0]]
    uniform = UniformLevelSampler(levels, seed=0)
    counts = {level_id: 0 for level_id in ids}
    for _ in range(2000):
        counts[uniform.sample(1)[0]] += 1
    assert min(counts.values()) > 200
    made = make_sampler(levels, mode="single")
    assert isinstance(made, SingleLevelSampler)


def test_adaptive_sampler_prioritises_weak_levels(catalog: LevelCatalog) -> None:
    levels = list(catalog.train_levels[:4])
    ids = [level.id for level in levels]
    config = CurriculumConfig(exploration_fraction=0.0, weakness_scale=6.0)
    sampler = AdaptiveLevelSampler(levels, seed=0, config=config)
    # the first level is mastered, the second is terrible
    for step in range(20):
        sampler.update(
            ids[0], success=True, progress=1.0, episode_return=40.0, step=step * 100
        )
        sampler.update(
            ids[1], success=False, progress=0.05, episode_return=-8.0, step=step * 100 + 50
        )
    counts = {level_id: 0 for level_id in ids}
    for _ in range(4000):
        counts[sampler.sample(1)[0]] += 1
    assert counts[ids[1]] > counts[ids[0]]
    assert sampler.performance[ids[0]].attempts >= 20
    summary = summarise_performance(sampler)
    assert summary and summary[0]["level_id"] in ids


def test_curriculum_config_round_trip() -> None:
    config = CurriculumConfig(weakness_scale=4.5, novelty_bonus=2.5)
    restored = CurriculumConfig.from_dict(config.to_dict())
    assert restored.weakness_scale == 4.5
    assert restored.novelty_bonus == 2.5
    with pytest.raises(ValueError):
        CurriculumConfig.from_dict({"nope": 1.0})


def test_sampled_actions_stay_inside_the_control_range() -> None:
    """PPO must train on the action the environment actually executes.

    The env clips actions to ``[-1, 1]``.  Before this test existed the rollout stored the raw
    Gaussian sample and its log-probability, so ~25 % of action dimensions in a trained
    rollout were scored against values that were never applied.
    """

    model = ActorCritic(OBSERVATION_DIM, ACTION_DIM, hidden_dim=32, seed=5, initial_log_std=0.0)
    rng = np.random.default_rng(1)
    observations = rng.normal(0, 3.0, (64, OBSERVATION_DIM))
    actions, log_probs, _ = model.sample(observations, rng)
    assert np.all(np.abs(actions) <= 1.0)
    # The stored log-probability is that of the stored (clipped) action.
    recomputed, _, _ = model.evaluate_actions(observations, actions)
    np.testing.assert_allclose(recomputed, log_probs, rtol=1e-9, atol=1e-12)
    # With a wide policy some samples would have left the range unclipped.
    raw = model.action_mean(observations) + np.exp(model.log_std) * rng.standard_normal(
        actions.shape
    )
    assert np.mean(np.abs(raw) > 1.0) > 0.0


def test_benchmark_simulation_measures_the_requested_worker_layout() -> None:
    """A benchmark that ignores ``workers`` cannot justify the training defaults."""

    from parkour_lab.training.vector import benchmark_simulation

    catalog = LevelCatalog.load()
    levels = ["level-01-basics", "level-02-step-up"]
    serial = benchmark_simulation(catalog, levels=levels, num_envs=2, steps=8, warmup=2, workers=1)
    assert serial["workers"] == 1
    assert serial["num_envs"] == 2
    assert serial["transitions"] == 16
    assert serial["steps_per_second"] > 0
    parallel = benchmark_simulation(
        catalog, levels=levels, num_envs=2, steps=8, warmup=2, workers=2
    )
    assert parallel["workers"] == 2


def test_finished_episode_is_labelled_with_the_level_it_ran(catalog: LevelCatalog) -> None:
    """A level assigned for the *next* reset must not relabel the episode that just ended."""

    from parkour_lab.env import EnvConfig
    from parkour_lab.training.vector import SerialVecEnv

    vec = SerialVecEnv(
        catalog,
        ["level-01-basics"],
        env_config=EnvConfig(level_id="level-01-basics", max_episode_seconds=0.2),
        seeds=[0],
    )
    try:
        vec.reset()
        # Assign a different level while the episode is still running.
        vec.set_levels(["level-03-small-hops"])
        rng = np.random.default_rng(0)
        record = None
        for _ in range(200):
            _obs, _rew, terminated, truncated, infos = vec.step(
                rng.uniform(-1.0, 1.0, (1, vec.action_dim))
            )
            if terminated[0] or truncated[0]:
                record = infos[0]["episode"]
                break
        assert record is not None
        assert record["level_id"] == "level-01-basics"
        assert record["assigned_level"] == "level-03-small-hops"
    finally:
        vec.close()


def test_gae_bootstraps_a_truncated_episode() -> None:
    """A time limit is not a terminal state: V(s_final) still counts."""

    rewards = np.array([1.0, 1.0])
    values = np.array([0.5, 0.5])
    dones = np.array([0.0, 1.0])
    bootstrap = np.array([0.0, 2.0])
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.5,
        gae_lambda=0.5,
        last_value=0.0,
        bootstrap=bootstrap,
    )
    # hand-computed: delta_1 = 1 + 0 + 2.0 - 0.5 = 2.5; delta_0 = 1 + 0.25 + 0 - 0.5 = 0.75
    np.testing.assert_allclose(advantages, [0.75 + 0.25 * 2.5, 2.5], atol=1e-12)
    np.testing.assert_allclose(returns, advantages + values, atol=1e-12)
    # Without the bootstrap the truncated step would be worth exactly its reward.
    plain, _ = compute_gae(rewards, values, dones, gamma=0.5, gae_lambda=0.5, last_value=0.0)
    assert plain[1] < advantages[1]
    assert plain[1] == pytest.approx(0.5)
