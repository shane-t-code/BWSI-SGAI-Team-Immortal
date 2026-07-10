"""Adversarial image generation training script.

Trains an RL agent to generate harder-to-classify images by learning optimal
transformations that target the CNN's decision boundaries while following
the 5 rules for solvable, realistic image generation.

Train via:
    python main.py train --type adversarial
    python main.py train --type adversarial -e 500 -s 7
    python main.py train --help

Hyperparameters to tune:
    NUM_EPISODES        — total training episodes (default: 500)
    BATCH_SIZE          — transitions per gradient step (default: 32)
    BUFFER_CAPACITY     — max replay buffer size (default: 2000)
    EPSILON_DECAY       — episodes over which epsilon decays (default: 250)
    TARGET_UPDATE_FREQ  — episodes between target net hard updates (default: 10)
    GAMMA               — discount factor (default: 0.95)
    LEARNING_RATE       — Adam step size (default: 1e-3)
    MAX_DIFFICULTY      — maximum allowed difficulty (default: 0.8)
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

# When run directly as a script (python training/train_adversarial.py), Python adds
# training/ to sys.path — not the project root. This block ensures the root
# is on sys.path so local packages (game_interfaces, gameplay, models) import.
# Has no effect when imported as a module (main.py adds the root itself).
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from gameplay.constants import DATA_DIR  # noqa: E402
from models.CNNClassifier import CNNClassifier  # noqa: E402
from models.DQNAgent import DQNAgent  # noqa: E402
from game_interfaces.adversarial_env import AdversarialEnv  # noqa: E402

# -----------------------------------------------------------------------
# Hyperparameters — edit these to experiment
# -----------------------------------------------------------------------
NUM_EPISODES: int = 500
BATCH_SIZE: int = 32
BUFFER_CAPACITY: int = 2000
EPSILON_DECAY: int = 250  # episodes over which epsilon decays linearly to epsilon_end
TARGET_UPDATE_FREQ: int = 10
GAMMA: float = 0.95
LEARNING_RATE: float = 1e-3
MAX_DIFFICULTY: float = 0.8  # Maximum difficulty to maintain solvability (Rule 1)

# Log a summary line every N episodes
LOG_INTERVAL: int = 25

# Seed for reproducibility (set to None to disable)
SEED: int | None = 42

ADVERSARIAL_LOG_DIR: str = "training/Adversarial-logs"
ADVERSARIAL_MODEL_PATH: str = "models/weights/adversarial_agent.pth"
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


def make_env(data_dir: str, seed: int | None = None) -> AdversarialEnv:
    """Build a fresh AdversarialEnv from the data directory.

    Args:
        data_dir: Path to the directory containing humanoid image files.
        seed: Random seed for reproducible training. Pass None to disable seeding.

    Returns:
        A fully initialized AdversarialEnv.
    """
    classifier = CNNClassifier(img_data_root=data_dir)
    return AdversarialEnv(
        classifier=classifier,
        img_data_root=data_dir,
        max_difficulty=MAX_DIFFICULTY,
    )


def _save_adversarial_plots(
    episode_rewards: list[float],
    epsilon_log: list[float],
    difficulty_log: list[float],
    log_dir: str,
) -> None:
    """Save adversarial training curves to log_dir.

    Args:
        episode_rewards: Total reward per episode, in order.
        epsilon_log: Epsilon value at the end of each episode, in order.
        difficulty_log: Average difficulty (1 - max_prob) per episode.
        log_dir: Directory to write plot files into.
    """
    os.makedirs(log_dir, exist_ok=True)
    episodes_range = range(1, len(episode_rewards) + 1)
    window = min(25, len(episode_rewards))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

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
    ax1.set_title("Episode Reward (Higher = Better Difficulty)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Middle: epsilon decay ─────────────────────────────────────────────
    ax2.plot(episodes_range, epsilon_log, color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Exploration Rate (Epsilon Decay)")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # ── Bottom: difficulty progression ────────────────────────────────────
    ax3.plot(episodes_range, difficulty_log, color="green", linewidth=1.5)
    ax3.axhline(MAX_DIFFICULTY, color="red", linestyle="--", alpha=0.5, label="Max difficulty")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Difficulty (1 - max_prob)")
    ax3.set_title("Classification Difficulty Over Time")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Adversarial Image Generation Training", fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(log_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {curves_path}")


def train(
    data_dir: str = DATA_DIR,
    episodes: int = NUM_EPISODES,
    seed: int | None = SEED,
    save_path: str = ADVERSARIAL_MODEL_PATH,
    log_dir: str = ADVERSARIAL_LOG_DIR,
) -> None:
    """Runs the full adversarial DQN training loop and saves the best weights.

    Episode loop:
        1. Reset environment — get initial observation (random image).
        2. Agent selects transformation type.
        3. Agent selects intensity level.
        4. Apply transformation and evaluate classification difficulty.
        5. Receive reward based on difficulty metrics.
        6. Train agent on experience.
        7. Track best performing model.

    Args:
        data_dir: Path to the directory containing humanoid image files.
        episodes: Total number of training episodes.
        seed: Random seed for reproducibility. Pass None to disable seeding.
        save_path: Path to write the trained policy_net weights (.pth).
        log_dir: Directory to write training plots into (default: Adversarial-logs/).
    """
    if seed is not None:
        set_seed(seed)

    env = make_env(data_dir, seed=seed)

    agent = DQNAgent(
        obs_size=env.observation_space.shape[0],
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
    difficulty_log: list[float] = []

    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        done = False

        # Track original features for comparison
        original_max_prob = info["original_features"]["max_prob"]

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _truncated, info = env.step(action)

            # Push transition to replay buffer
            agent.replay_buffer.push(obs, action, reward, next_obs, done or _truncated)

            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss

            obs = next_obs
            episode_reward += reward
            steps += 1

        # End-of-episode bookkeeping
        target_updated = agent.end_episode()
        episode_rewards.append(episode_reward)
        epsilon_log.append(agent.epsilon)

        # Track difficulty (1 - max_prob, higher = harder)
        if "features" in info:
            difficulty = 1.0 - info["features"]["max_prob"]
            difficulty_log.append(difficulty)

        # Save best weights
        if episode_reward > best_reward and agent.epsilon < 0.5:
            best_reward = episode_reward
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            agent.save(save_path)
            weights_saved = True
            print(
                f"  -> new best {episode_reward:+.2f} at ep {episode} "
                f"(eps={agent.epsilon:.2f}) -- saved to {save_path}"
            )

        if episode % LOG_INTERVAL == 0:
            recent = episode_rewards[-LOG_INTERVAL:]
            avg = sum(recent) / len(recent)
            avg_loss = episode_loss / steps if steps > 0 else 0.0
            avg_difficulty = sum(difficulty_log[-LOG_INTERVAL:]) / len(difficulty_log[-LOG_INTERVAL:]) if difficulty_log else 0.0
            target_flag = " [target updated]" if target_updated else ""
            print(
                f"Episode {episode:4d}/{episodes}  "
                f"reward: {episode_reward:+.2f}  "
                f"avg(last {LOG_INTERVAL}): {avg:+.2f}  "
                f"loss(ep): {avg_loss:.4f}  "
                f"epsilon: {agent.epsilon:.3f}  "
                f"difficulty: {avg_difficulty:.3f}  "
                f"steps: {steps}"
                f"{target_flag}"
            )

    if weights_saved:
        print(f"\nTraining complete. Best episode reward: {best_reward:+.2f}")
        print(f"Weights saved to: {save_path}")
    else:
        print(
            "\nTraining complete. No weights saved — epsilon never dropped below 0.5."
        )
        print("Try increasing NUM_EPISODES or decreasing EPSILON_DECAY.")

    _save_adversarial_plots(episode_rewards, epsilon_log, difficulty_log, log_dir)


if __name__ == "__main__":
    # Allow running this script directly from the training/ directory:
    #   python train_adversarial.py
    # Sets cwd to project root so relative paths (DATA_DIR="data") resolve
    # correctly. sys.path is already fixed at import time (see top of file).
    # To pass flags (-e, -s, -d, -o), use the main CLI instead:
    #   python main.py train --type adversarial
    os.chdir(_project_root)
    train()
