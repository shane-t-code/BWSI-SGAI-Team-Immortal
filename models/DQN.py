"""Deep Q-Network (DQN) model for the BWSI SGAI project."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """A Q-network that maps observations to action-value estimates (Q-values).

    Unlike a policy network, this network does NOT output probabilities — there
    is no softmax at the end. Instead, each output is a raw Q-value: the expected
    cumulative discounted reward for taking that action from the current state.
    The agent acts by picking the action with the highest Q-value (argmax).

    Architecture:
        obs_size -> Linear(hidden_size) -> ReLU
                 -> Linear(hidden_size) -> ReLU
                 -> Linear(num_actions)

    Args:
        obs_size: Number of features in the flattened observation vector.
        num_actions: Number of discrete actions the agent can take.
        hidden_size: Width of each hidden layer. Defaults to 128.

    Example:
        >>> net = DQN(obs_size=64, num_actions=5)
        >>> obs = torch.zeros(32, 64)  # batch of 32 observations
        >>> q_values = net(obs)        # shape: (32, 5)
    """

    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 128) -> None:
        """Initializes the DQN with two hidden layers.

        Args:
            obs_size: Dimension of the input observation vector.
            num_actions: Number of discrete actions (output dimension).
            hidden_size: Number of neurons in each hidden layer. 128 is a
                reasonable default for this project — large enough to capture
                the relationships between state features (humanoid types,
                ambulance capacity, shift time, etc.) without being so large
                that it overfits or slows down training on a small obs vector.
                If your observation space grows significantly, try 256 or 512.
        """
        super().__init__()

        # nn.Sequential chains layers in order: input flows through each layer
        # top-to-bottom. ReLU is a simple non-linearity: output = max(0, x).
        # Two hidden layers give the network enough depth to learn non-linear
        # combinations of state features without being hard to train.
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            # No activation here — raw Q-values, not probabilities.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Q-values for each action given a batch of observations.

        Args:
            x: Observation tensor of shape (batch_size, obs_size). Each row
                is one flattened observation from the environment.

        Returns:
            Q-value tensor of shape (batch_size, num_actions). Entry [i, a]
            is the estimated return for taking action a in observation i.
            No softmax is applied — these are raw scores, not probabilities.
        """
        # x must be 2D: (batch_size, obs_size). If you get a shape error below,
        # check that you're not passing a 1D tensor — use x.unsqueeze(0) for a
        # single observation.
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch, obs_size), got shape {tuple(x.shape)}"
            )
        return self.network(x)
