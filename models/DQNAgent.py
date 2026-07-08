"""DQN agent: ReplayBuffer + DQNAgent for the BWSI SGAI project."""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gameplay.constants import OBS_SIZE
from models.DQN import DQN


class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Stores (state, action, reward, next_state, done) tuples and samples
    random mini-batches for off-policy training. The deque automatically
    evicts the oldest experience when capacity is exceeded.

    Attributes:
        buffer: Underlying deque that stores transitions.

    Example:
        >>> buf = ReplayBuffer(capacity=5000)
        >>> buf.push(obs, action, reward, next_obs, done)
        >>> batch = buf.sample(32)
    """

    buffer: deque
    rng: random.Random

    def __init__(self, capacity: int = 5000, seed: int | None = None) -> None:
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            seed: Optional random seed for reproducible mini-batch sampling.
        """
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the buffer.

        Args:
            state: Normalized observation array as returned by ``env.step()``
                (already in ``[0, 1]`` per slot — ``build_obs()`` normalizes
                before the obs reaches the buffer).
            action: Integer action index taken.
            reward: Scalar reward received.
            next_state: Normalized next observation array.
            done: True if the episode ended after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones), each
            a numpy array with batch_size rows.
        """
        batch = self.rng.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and a target network.

    Maintains two networks — policy_net (updated every step) and target_net
    (hard-copied from policy_net every target_update_freq episodes). This
    stabilizes training: the Bellman target is computed from target_net,
    so it doesn't move on every gradient step.

    Training uses experience replay: transitions are stored in a ReplayBuffer
    and sampled randomly, breaking the temporal correlations that would
    destabilize a naive online update.

    Epsilon-greedy schedule:
        - Starts at epsilon_start (1.0) — fully random exploration
        - Decays linearly to epsilon_end (0.05) over epsilon_decay episodes
        - Stays at epsilon_end thereafter

    Args:
        obs_size: Dimension of the flattened observation vector (15 for this project).
        num_actions: Number of discrete actions (4 for this project).
        hidden_size: Width of each hidden layer in the DQN. Defaults to 128.
        lr: Adam learning rate. Defaults to 1e-3.
        gamma: Discount factor for future rewards. Defaults to 0.99.
        epsilon_start: Initial exploration rate. Defaults to 1.0.
        epsilon_end: Minimum exploration rate. Defaults to 0.05.
        epsilon_decay: Number of episodes over which epsilon decays linearly.
            Defaults to 300.
        buffer_capacity: Max transitions in the replay buffer. Defaults to 5000.
        batch_size: Mini-batch size for each train_step call. Defaults to 32.
        target_update_freq: Episodes between hard target-net updates. Defaults to 20.
    """

    num_actions: int
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: int
    batch_size: int
    target_update_freq: int
    epsilon: float
    episode: int
    device: torch.device
    policy_net: DQN
    target_net: DQN
    optimizer: optim.Adam
    loss_fn: nn.SmoothL1Loss
    replay_buffer: ReplayBuffer
    rng: random.Random

    # Max values used to normalize each observation slot independently.
    # These match the encoding in game_interfaces/obs_builder.py build_obs():
    #   slot 0      time_bin            max bin = 4
    #   slots 1-4   class_counts        already in [0,1] (÷ capacity in env)
    #   slots 5-8   humanoid_class      one-hot — already in {0, 1}
    #   slot 9      humanoid_confidence max bin = 2 → stored as bin/2, already [0,1]
    #   slots 10-13 doable_actions      binary — already in {0, 1}
    #   slot 14     humanoid_value_bin  stored as bin/4, already [0,1]
    # All slots land in [0,1] after env normalization, so _normalize_obs is
    # a no-op safety pass (divide by 1.0 everywhere). Kept explicit so
    # this shows where normalization lives — extend it if the obs space changes.
    # NOTE: _OBS_MAX is set as an instance attribute in __init__, sized to obs_size.

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        num_actions: int = 4,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 300,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        target_update_freq: int = 20,
        seed: int | None = None,
    ) -> None:
        if epsilon_decay <= 0:
            raise ValueError(
                f"epsilon_decay must be > 0, got {epsilon_decay}. "
                "epsilon_decay=0 causes ZeroDivisionError in end_episode()."
            )
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        if target_update_freq <= 1:
            raise ValueError(
                f"target_update_freq must be >= 2, got {target_update_freq}. "
                "freq=0 causes ZeroDivisionError in end_episode(); "
                "freq=1 copies the target net every episode, defeating stable Bellman targets. "
                "Recommended range: 10–50 episodes."
            )

        self.epsilon = epsilon_start
        self.episode = 0  # incremented by end_episode()
        self.rng = random.Random(seed)  # seeded RNG for epsilon-greedy exploration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Two networks: policy_net is trained every step; target_net is a
        # periodic hard copy used to compute stable Bellman targets.
        self.policy_net = DQN(obs_size, num_actions, hidden_size).to(self.device)
        self.target_net = DQN(obs_size, num_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target_net is never trained directly

        # _OBS_MAX: per-slot normalization ceiling — all 1.0 since env already normalizes to [0,1].
        # Parameterized by obs_size so it scales automatically when the observation vector grows.
        self._OBS_MAX = np.ones(obs_size, dtype=np.float32)

        assert (
            self._OBS_MAX > 0
        ).all(), "_OBS_MAX must have all positive values — a zero entry causes nan in normalize_obs()"

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # SmoothL1Loss (Huber loss) clips gradients for large TD errors, preventing
        # the exploding-gradient instability that MSELoss causes when Q-values are
        # large early in training. This matches the original DQN paper (Mnih 2015).
        self.loss_fn = nn.SmoothL1Loss()

        # Pass a derived seed to ReplayBuffer so mini-batch sampling is also reproducible.
        buf_seed = (seed + 1) if seed is not None else None
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=buf_seed)

    # ------------------------------------------------------------------
    # Observation normalization
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalizes a raw observation vector to [0, 1] per slot.

        The environment already outputs values in [0, 1], so this is a
        safety pass. If you add observation features, _OBS_MAX scales
        automatically (it is sized to obs_size in __init__).

        Args:
            obs: Raw float32 observation array of shape (obs_size,).

        Returns:
            Float32 array of the same shape, each element in [0, 1].
        """
        return (obs / self._OBS_MAX).astype(np.float32)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        greedy: bool = False,
        available: list[bool] | None = None,
    ) -> int:
        """Selects an action using epsilon-greedy policy.

        Args:
            obs: Observation array of shape (obs_size,). Will be normalized
                internally before passing to the network.
            greedy: If True, always pick the argmax (no exploration). Use
                this at inference time.
            available: Optional boolean list of length ``num_actions``. When
                provided, unavailable actions are masked to ``-inf`` before
                argmax so they are never selected. Ignored during random
                exploration.

        Returns:
            Integer action index in [0, num_actions).
        """
        if not greedy and self.rng.random() < self.epsilon:
            if available is not None:
                choices = [i for i, a in enumerate(available) if a]
                if choices:
                    return self.rng.choice(choices)
            return self.rng.randrange(self.num_actions)

        norm_obs = self.normalize_obs(obs)
        state_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)  # (1, obs_size) for the network

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # shape: (1, num_actions)

        if available is not None:
            mask = torch.tensor(
                [0.0 if a else float("-inf") for a in available],
                dtype=torch.float32,
                device=self.device,
            )
            q_values = q_values + mask

        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """Samples a mini-batch and performs one gradient update on policy_net.

        Skips silently if the buffer doesn't have enough transitions yet.

        Bellman target:
            Q_target(s, a) = r  +  γ * max_a' Q_target(s')   if not done
            Q_target(s, a) = r                                  if done

        Huber loss (SmoothL1) between predicted Q(s, a) and Q_target(s, a) is
        minimized via a single Adam step on policy_net only — target_net is frozen.

        Returns:
            Scalar loss value, or None if the buffer was too small to sample.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.policy_net.train()  # enable Dropout/BatchNorm training mode during updates

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Normalize observations
        states = np.array([self.normalize_obs(s) for s in states], dtype=np.float32)
        next_states = np.array(
            [self.normalize_obs(s) for s in next_states], dtype=np.float32
        )

        # Convert to tensors
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Q(s, a) from policy_net — gather the Q-value for the action that was taken
        q_pred = self.policy_net(states_t)  # (batch, num_actions)
        q_pred = q_pred.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (batch,)

        # Q_target(s', a') from target_net — Bellman equation
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1).values  # (batch,)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()  # clear old gradients before backward pass
        loss.backward()
        self.optimizer.step()
        self.policy_net.eval()  # restore eval mode after update step

        return loss.item()

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self) -> bool:
        """Performs all end-of-episode bookkeeping: epsilon decay + target update.

        Call exactly once after each episode finishes. Handles both:
          1. Epsilon decay — linear from epsilon_start to epsilon_end over
             epsilon_decay episodes, then fixed at epsilon_end.
          2. Target net hard update — copies policy_net → target_net every
             target_update_freq episodes. The hard copy gives the Bellman
             target a stable reference: if target_net moved every step,
             the prediction and target would chase each other (moving-target
             problem), destabilizing training.

        Returns:
            True if the target network was updated this episode, False otherwise.
            Useful for logging.
        """
        self.episode += 1

        # Linear epsilon decay
        progress = min(1.0, self.episode / self.epsilon_decay)
        self.epsilon = self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

        # Hard update target net every target_update_freq episodes
        if self.episode % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves policy_net weights to disk.

        Creates the parent directory if it does not exist.

        Args:
            path: File path to write (e.g. 'models/weights/baselineRL.pth').
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Loads policy_net weights from disk and syncs target_net.

        Args:
            path: File path to read.
        """
        load_kwargs: dict = {"map_location": self.device, "weights_only": True}
        state_dict = torch.load(path, **load_kwargs)  # nosec B614
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()
