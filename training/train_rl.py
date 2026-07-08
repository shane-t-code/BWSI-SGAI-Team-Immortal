"""DQN training script for the BWSI SGAI project.

Trains a DQNAgent on the game environment using experience replay and
a target network. Produces a light baseline — enough for non-random behavior.
Extend it for your own experiments.

Train via:
    python main.py train --type rl
    python main.py train --type rl -e 1000 -s 7
    python main.py train --help

Note on live progress output:
    Python buffers stdout when it is not connected to a terminal (e.g. when
    running in the background or redirecting output to a file). This means
    the per-episode log lines may not appear until the buffer fills or
    training completes. To force unbuffered output and see progress in
    real time, use the -u flag:
        python -u -m training.train_rl
    or via main.py:
        python -u main.py train --type rl

Hyperparameters to tune:
    NUM_EPISODES        — total training episodes (default: 2500)
    BATCH_SIZE          — transitions per gradient step (default: 32)
    BUFFER_CAPACITY     — max replay buffer size (default: 5000)
    EPSILON_DECAY       — episodes over which epsilon decays (default: 1250)
    TARGET_UPDATE_FREQ  — episodes between target net hard updates (default: 20)
    GAMMA               — discount factor (default: 0.99)
    LEARNING_RATE       — Adam step size (default: 1e-3)

Accuracy improvements to try:
    - More episodes: default is 2500; try 500–1000 for a quick experiment run
    - Tune GAMMA: lower values (0.9) favor immediate rewards; higher (0.99) plan ahead
    - Tune EPSILON_DECAY: slower decay keeps exploration longer on small datasets
    - Reward shaping: modify TrainEnv.step() — the step reward is computed there
      as the delta in cumulative reward since the last step
"""

import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

# When run directly as a script (python training/train_rl.py), Python adds
# training/ to sys.path — not the project root. This block ensures the root
# is on sys.path so local packages (game_interfaces, gameplay, models) import.
# Has no effect when imported as a module (main.py adds the root itself).
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from game_interfaces.data_parser import DataParser  # noqa: E402
from game_interfaces.train_env import TrainEnv  # noqa: E402
from gameplay.constants import (  # noqa: E402
    AMBULANCE_CAPACITY,
    DATA_DIR,
    RL_MODEL_PATH,
    SHIFT_LENGTH,
)
from gameplay.scorekeeper import ScoreKeeper  # noqa: E402
from models.CNNClassifier import CNNClassifier  # noqa: E402
from models.DQNAgent import DQNAgent  # noqa: E402

# -----------------------------------------------------------------------
# Hyperparameters — edit these to experiment
# -----------------------------------------------------------------------
NUM_EPISODES: int = 2500
BATCH_SIZE: int = 32
BUFFER_CAPACITY: int = 5000
EPSILON_DECAY: int = 1250  # episodes over which epsilon decays linearly to epsilon_end
TARGET_UPDATE_FREQ: int = 20
GAMMA: float = 0.99
LEARNING_RATE: float = 1e-3

# Log a summary line every N episodes
LOG_INTERVAL: int = 50

# Seed for reproducibility (set to None to disable)
SEED: int | None = 42

RL_LOG_DIR: str = "training/RL-logs"
# -----------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(data_dir: str, seed: int | None = None) -> TrainEnv:
    """Build a fresh TrainEnv from the data directory.

    Args:
        data_dir: Path to the directory containing ``consolidated_metadata.csv``
            and the humanoid image files.
        seed: Random seed for reproducible training — passed to DataParser
            and ScoreKeeper. Pass ``None`` to disable seeding.

    Returns:
        A fully initialized :class:`~game_interfaces.train_env.TrainEnv`.
    """
    data_parser = DataParser(data_dir=data_dir, seed=seed)
    scorekeeper = ScoreKeeper(
        shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY, seed=seed
    )
    classifier = CNNClassifier(img_data_root=data_dir)
    return TrainEnv(
        root=None,
        w=0,
        h=0,
        data_parser=data_parser,
        scorekeeper=scorekeeper,
        classifier=classifier,
        display=False,
    )


def _save_rl_plots(
    episode_rewards: list[float],
    epsilon_log: list[float],
    log_dir: str,
) -> None:
    """Save RL training curve and epsilon decay plots to log_dir.

    Args:
        episode_rewards: Total reward per episode, in order.
        epsilon_log: Epsilon value at the end of each episode, in order.
        log_dir: Directory to write plot files into.
    """
    os.makedirs(log_dir, exist_ok=True)
    episodes_range = range(1, len(episode_rewards) + 1)
    window = min(50, len(episode_rewards))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # ── Top: episode reward + moving average ──────────────────────────────
    ax1.plot(
        episodes_range,
        episode_rewards,
        alpha=0.4,
        color="steelblue",
        label="Episode reward",
    )
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        ma_range = range(window, len(episode_rewards) + 1)
        ax1.plot(
            ma_range,
            moving_avg,
            color="steelblue",
            linewidth=2,
            label=f"{window}-ep moving avg",
        )
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Bottom: epsilon decay ─────────────────────────────────────────────
    ax2.plot(episodes_range, epsilon_log, color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Exploration Rate (Epsilon Decay)")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("DQN Training Curves", fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(log_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {curves_path}")


def train(
    data_dir: str = DATA_DIR,
    episodes: int = NUM_EPISODES,
    seed: int | None = SEED,
    save_path: str = RL_MODEL_PATH,
    log_dir: str = RL_LOG_DIR,
) -> None:
    """Runs the full DQN training loop and saves the best weights.

    Episode loop:
        1. Reset environment — get initial observation.
        2. Until episode ends:
           a. Select action (epsilon-greedy).
           b. Step environment — get (next_obs, reward, done, truncated, info).
           c. Push transition to replay buffer.
           d. Run one train_step() on the agent.
        3. Call agent.end_episode() — decays epsilon, optionally hard-updates
           target net every TARGET_UPDATE_FREQ episodes.
        4. Track episode reward; save best weights so far.

    Args:
        data_dir: Path to the directory containing consolidated_metadata.csv
            and the humanoid image files.
        episodes: Total number of training episodes.
        seed: Random seed for reproducibility. Pass None to disable seeding.
        save_path: Path to write the trained policy_net weights (.pth).
        log_dir: Directory to write training plots into (default: RL-logs/).
    """
    if seed is not None:
        set_seed(seed)

    env = make_env(data_dir, seed=seed)

    agent = DQNAgent(
        obs_size=env.obs_size,
        num_actions=int(env.action_space.n),
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    best_reward = float("-inf")
    weights_saved = False
    episode_rewards: list[float] = []
    epsilon_log: list[float] = []

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _truncated, _info = env.step(action)

            # Push raw obs — DQNAgent.train_step normalizes internally
            # Use done or _truncated: truncated episodes are also terminal for
            # bootstrap purposes — without this, Q-targets add γ·V(next_obs)
            # on a state the agent will never actually continue from.
            agent.replay_buffer.push(obs, action, reward, next_obs, done or _truncated)

            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss

            obs = next_obs
            episode_reward += reward
            steps += 1

        # End-of-episode bookkeeping: epsilon decay + target net update
        target_updated = agent.end_episode()
        episode_rewards.append(episode_reward)
        epsilon_log.append(agent.epsilon)

        # Save the weights from the single highest-reward episode seen so far.
        # Watch out: early episodes run with high epsilon (lots of random actions),
        # so a lucky random episode can set best_reward before the agent has learned
        # anything useful. The epsilon value printed on save tells you whether this
        # happened — a save at ε > 0.5 means the "best" model is mostly random.
        if episode_reward > best_reward and agent.epsilon < 0.5:
            best_reward = episode_reward
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            agent.save(save_path)
            weights_saved = True
            print(
                f"  -> new best {episode_reward:+.1f} at ep {episode} "
                f"(eps={agent.epsilon:.2f}) -- saved to {save_path}"
            )

        if episode % LOG_INTERVAL == 0:
            recent = episode_rewards[-LOG_INTERVAL:]
            avg = sum(recent) / len(recent)
            # avg_loss is per-step average for this episode only, not across the interval
            avg_loss = episode_loss / steps if steps > 0 else 0.0
            target_flag = " [target updated]" if target_updated else ""
            print(
                f"Episode {episode:4d}/{episodes}  "
                f"reward: {episode_reward:+.1f}  "
                f"avg(last {LOG_INTERVAL}): {avg:+.1f}  "
                f"loss(ep): {avg_loss:.4f}  "
                f"epsilon: {agent.epsilon:.3f}  "
                f"steps: {steps}"
                f"{target_flag}"
            )

    if weights_saved:
        print(f"\nTraining complete. Best episode reward: {best_reward:+.1f}")
        print(f"Weights saved to: {save_path}")
    else:
        print(
            "\nTraining complete. No weights saved — epsilon never dropped below 0.5."
        )
        print("Try increasing NUM_EPISODES or decreasing EPSILON_DECAY.")

    _save_rl_plots(episode_rewards, epsilon_log, log_dir)


if __name__ == "__main__":
    # Allow running this script directly from the training/ directory:
    #   python train_rl.py
    # Sets cwd to project root so relative paths (DATA_DIR="data") resolve
    # correctly. sys.path is already fixed at import time (see top of file).
    # To pass flags (-e, -s, -d, -o), use the main CLI instead:
    #   python main.py train --type rl
    os.chdir(_project_root)
    train()
