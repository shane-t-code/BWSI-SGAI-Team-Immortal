"""Tests for models/DQNAgent.py.

COV-19: train_step returns None when buffer too small, select_action greedy range,
        epsilon decay to minimum, and save/load round-trip preserves Q-values.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gameplay.constants import OBS_SIZE
from models.DQNAgent import DQNAgent

_NUM_ACTIONS = 4

# Value-bin encoding: slot [14] is value_bin / 4, so valid values are {0, 0.25, 0.5, 0.75, 1.0}
_VALUE_BIN_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture
def agent() -> DQNAgent:
    return DQNAgent(obs_size=OBS_SIZE, num_actions=_NUM_ACTIONS, epsilon_decay=5)


def _random_obs() -> np.ndarray:
    """Build a valid 15-slot observation respecting slot semantics.

    Slot layout:
        [0]     time_bin            uniform in [0, 1]
        [1-4]   class_counts        uniform in [0, 1] each
        [5-8]   humanoid_class      one-hot (exactly one slot = 1.0)
        [9]     humanoid_confidence one of {0.0, 0.5, 1.0}
        [10-13] doable_actions      binary {0.0, 1.0} each
        [14]    humanoid_value_bin  one of {0.0, 0.25, 0.5, 0.75, 1.0}
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = np.random.uniform(0.0, 1.0)  # time_bin
    obs[1:5] = np.random.uniform(0.0, 1.0, size=4)  # class_counts
    obs[5 + np.random.randint(4)] = 1.0  # humanoid_class one-hot
    obs[9] = np.random.choice([0.0, 0.5, 1.0])  # humanoid_confidence
    obs[10:14] = np.random.randint(0, 2, size=4).astype(np.float32)  # doable_actions
    obs[14] = np.random.choice(_VALUE_BIN_VALUES)  # humanoid_value_bin
    return obs


# ---------------------------------------------------------------------------
# COV-19a — train_step returns None when buffer is too small
# ---------------------------------------------------------------------------


def test_train_step_returns_none_when_buffer_too_small(agent):
    """train_step() returns None when the replay buffer has fewer entries than batch_size."""
    # Push fewer transitions than batch_size (32)
    obs = _random_obs()
    for _ in range(5):
        agent.replay_buffer.push(obs, 0, 0.0, obs, False)
    result = agent.train_step()
    assert result is None


# ---------------------------------------------------------------------------
# COV-19b — select_action greedy returns valid index
# ---------------------------------------------------------------------------


def test_select_action_greedy_returns_valid_index(agent):
    """select_action(greedy=True) always returns an integer in [0, num_actions)."""
    obs = _random_obs()
    idx = agent.select_action(obs, greedy=True)
    assert isinstance(idx, int)
    assert 0 <= idx < _NUM_ACTIONS


def test_select_action_exploration_returns_valid_index(agent):
    """select_action(greedy=False) returns a valid action index via random exploration."""
    obs = _random_obs()
    idx = agent.select_action(obs, greedy=False)
    assert isinstance(idx, int)
    assert 0 <= idx < _NUM_ACTIONS


# ---------------------------------------------------------------------------
# COV-19c — epsilon decays to minimum
# ---------------------------------------------------------------------------


def test_epsilon_decays_to_minimum(agent):
    """After epsilon_decay episodes, epsilon equals epsilon_end."""
    for _ in range(agent.epsilon_decay + 10):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(agent.epsilon_end)


# ---------------------------------------------------------------------------
# COV-19d — save/load round-trip preserves Q-values
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_qvalues(agent, tmp_path):
    """Policy network Q-values are identical before save and after load into a new agent."""
    obs = _random_obs()
    obs_tensor = torch.tensor(
        agent.normalize_obs(obs), dtype=torch.float32, device=agent.device
    ).unsqueeze(0)

    agent.policy_net.eval()
    with torch.no_grad():
        q_before = agent.policy_net(obs_tensor).cpu().numpy()

    weights_path = str(tmp_path / "test_weights.pth")
    agent.save(weights_path)

    agent2 = DQNAgent(obs_size=OBS_SIZE, num_actions=_NUM_ACTIONS)
    agent2.load(weights_path)
    agent2.policy_net.eval()
    with torch.no_grad():
        q_after = agent2.policy_net(obs_tensor).cpu().numpy()

    np.testing.assert_allclose(q_before, q_after, atol=1e-6)


# ---------------------------------------------------------------------------
# COV-19e — obs size matches production network input layer
# ---------------------------------------------------------------------------


def test_obs_size_matches_network_input(agent):
    """OBS_SIZE constant matches the policy network's first-layer input features."""
    first_layer = agent.policy_net.network[0]
    assert len(_random_obs()) == first_layer.in_features == OBS_SIZE


# ---------------------------------------------------------------------------
# Validation guards — __init__ rejects bad hyperparameters
# ---------------------------------------------------------------------------


def test_init_raises_on_zero_epsilon_decay():
    """DQNAgent raises ValueError when epsilon_decay=0 (would cause ZeroDivisionError)."""
    with pytest.raises(ValueError, match="epsilon_decay"):
        DQNAgent(obs_size=OBS_SIZE, num_actions=_NUM_ACTIONS, epsilon_decay=0)


def test_init_raises_on_low_target_update_freq():
    """DQNAgent raises ValueError when target_update_freq <= 1."""
    with pytest.raises(ValueError, match="target_update_freq"):
        DQNAgent(obs_size=OBS_SIZE, num_actions=_NUM_ACTIONS, target_update_freq=1)


# ---------------------------------------------------------------------------
# select_action with available action mask
# ---------------------------------------------------------------------------


def test_select_action_mask_excludes_disabled_actions():
    """select_action() with available=[False, True, True, True] never returns action 0."""
    agent = DQNAgent(
        obs_size=OBS_SIZE, num_actions=_NUM_ACTIONS, epsilon_decay=5, seed=42
    )
    # Force pure exploration so the mask path is exercised
    agent.epsilon = 1.0
    available = [False, True, True, True]
    obs = _random_obs()
    results = {agent.select_action(obs, available=available) for _ in range(50)}
    assert 0 not in results, "disabled action 0 was selected despite mask"
    assert results.issubset({1, 2, 3})


def test_select_action_mask_single_available():
    """select_action() with only one available action always returns that action."""
    agent = DQNAgent(
        obs_size=OBS_SIZE, num_actions=_NUM_ACTIONS, epsilon_decay=5, seed=0
    )
    agent.epsilon = 1.0  # pure exploration
    available = [False, False, True, False]
    obs = _random_obs()
    for _ in range(20):
        assert agent.select_action(obs, available=available) == 2


# ---------------------------------------------------------------------------
# train_step — full path (buffer large enough to sample)
# ---------------------------------------------------------------------------


def test_train_step_returns_float_when_buffer_full():
    """train_step() returns a non-negative float when replay buffer >= batch_size."""
    agent = DQNAgent(
        obs_size=OBS_SIZE,
        num_actions=_NUM_ACTIONS,
        epsilon_decay=5,
        batch_size=2,
        seed=0,
    )
    obs = _random_obs()
    # Push exactly batch_size transitions to trigger training
    for action in range(2):
        agent.replay_buffer.push(obs, action, 1.0, obs, False)

    loss = agent.train_step()

    assert loss is not None, "train_step() returned None despite buffer >= batch_size"
    assert isinstance(loss, float)
    assert loss >= 0.0, f"loss {loss} is negative — unexpected for MSE/Huber"


def test_train_step_updates_policy_weights():
    """Policy net parameters change after a train_step() with sufficient buffer."""
    agent = DQNAgent(
        obs_size=OBS_SIZE,
        num_actions=_NUM_ACTIONS,
        epsilon_decay=5,
        batch_size=2,
        seed=1,
    )
    obs = _random_obs()
    agent.replay_buffer.push(obs, 0, 1.0, obs, False)
    agent.replay_buffer.push(obs, 1, -1.0, obs, True)

    # Snapshot weights before training
    before = [p.clone() for p in agent.policy_net.parameters()]

    agent.train_step()

    after = list(agent.policy_net.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert (
        changed
    ), "policy net weights unchanged after train_step() — gradient did not apply"
