"""SGAI game entry point.

Five game modes (GUI)
---------------------
  human        Pure human play — no machine assistance
  machine      Step-by-step machine play; press "Advance turn" to let the RL
               agent classify and act. Human watches.
  human_cnn    CNN identifies each humanoid; human decides the action
  human_rl     RL recommends an action; human decides (CNN runs internally
               to build the obs vector, but classification label is hidden)
  human_cnn_rl CNN identifies AND RL recommends; human decides the action

  NOTE: you can swap classifier and recommenders to other types.
  Default is CNN and Deep Q learning

Headless mode (no GUI — handled by _run_auto(), not _make_components())
------------------------------------------------------------------------
  auto         Fully headless — RL runs autonomously, prints final score

Training (utility — not a game mode)
-------------------------------------
  python main.py train --type rl   # train the DQN agent
  python main.py train --type cnn  # train the CNN image classifier
  See ``python main.py train --help`` for all options.

Usage
-----
  python main.py                            # human mode, default data dir
  python main.py -m auto                    # headless RL play
  python main.py -m auto --seed 42          # reproducible headless run
  python main.py -m human_rl --log          # human+RL recommendation with CSV logging
  python main.py -m human_cnn_rl            # human+CNN+RL full assist
  python main.py -m auto -d /path/to/data   # custom data directory
  python main.py train --type rl            # train DQN from scratch
  python main.py train --type rl -e 2500    # train DQN, 2500 episodes
  python main.py train --type cnn --seed 7  # reproducible CNN training
"""

import argparse
import random

import numpy as np
import torch

from game_interfaces.data_parser import DataParser
from gameplay.constants import (
    AMBULANCE_CAPACITY,
    CLASSIFIER_MODEL_PATH,
    CNN_CONFIDENCE_HIGH,
    CNN_CONFIDENCE_MID,
    DATA_DIR,
    RL_MODEL_PATH,
    SHIFT_LENGTH,
)
from gameplay.scorekeeper import ScoreKeeper
from models.ActionRecommender import ActionRecommender
from models.CNNClassifier import CNNClassifier
from models.HumanoidClassifier import HumanoidClassifier
from models.RLRecommender import RLRecommender
from training.train_cnn import NUM_EPOCHS
from training.train_cnn import train as train_cnn
from training.train_rl import NUM_EPISODES
from training.train_rl import train as train_rl
from ui.ui import UI


def _seed_all(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs.
    Args:
        seed: Integer seed value. Pass None to skip seeding.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_components(
    mode: str,
    data_dir: str,
) -> tuple[HumanoidClassifier | None, ActionRecommender | None, bool, bool]:
    """Factory — returns (classifier, recommender, auto_execute, show_classifier_label).

    The classifier is always passed to the recommender call site via the UI
    (or the auto loop). Recommenders do not own an internal classifier —
    probs are computed once by the caller and passed into recommend().

    Args:
        mode: One of 'human', 'machine', 'human_cnn', 'human_rl',
            'human_cnn_rl'. 'auto' is never passed here — it is intercepted
            by run() before this function is called and handled by _run_auto().
        data_dir: Root directory for image files, passed to classifiers.

    Returns:
        Tuple of (classifier, recommender, auto_execute, show_classifier_label).
        ``classifier`` and ``recommender`` are injectable interface instances
        (or None). ``auto_execute`` is True only in machine mode (the human
        presses Advance Turn; the RL agent picks the action automatically).
        ``show_classifier_label`` controls whether the CNN prediction is
        displayed to the player.
    """
    if mode == "human":
        return None, None, False, False

    if mode == "machine":
        return CNNClassifier(img_data_root=data_dir), RLRecommender(), True, True

    if mode == "human_cnn":
        return CNNClassifier(img_data_root=data_dir), None, False, True

    if mode == "human_rl":
        # Classifier runs internally to build the RL obs vector, but the
        # prediction label is hidden from the player — human sees only the
        # recommended action.
        return CNNClassifier(img_data_root=data_dir), RLRecommender(), False, False

    if mode == "human_cnn_rl":
        return CNNClassifier(img_data_root=data_dir), RLRecommender(), False, True

    raise ValueError(
        f"Unknown mode: {mode!r}. "
        "Valid GUI modes: 'human', 'machine', 'human_cnn', 'human_rl', 'human_cnn_rl'. "
        "For headless play, pass 'auto' to Main() — it is handled by _run_auto(), not this function."
    )


class Main:
    """Entry point for the SGAI game.

    Attributes:
        data_dir: Root directory for image files, passed to all components.
    """

    data_dir: str

    def __init__(
        self,
        mode: str,
        log: bool,
        data_dir: str = DATA_DIR,
        seed: int | None = None,
    ) -> None:
        """Initialize and run the game in the selected mode.

        Args:
            mode: Game mode — one of ``'human'``, ``'machine'``, ``'human_cnn'``,
                ``'human_rl'``, ``'human_cnn_rl'``, or ``'auto'``.
            log: If True, write a ``log.csv`` after the game ends.
            data_dir: Path to the directory containing images and metadata.
            seed: Random seed for reproducibility. None means no seeding (default).
                Useful with ``-m auto``.
        """
        _seed_all(seed)
        self.data_dir = data_dir
        data_parser = DataParser(self.data_dir, seed=seed)
        scorekeeper = ScoreKeeper(SHIFT_LENGTH, AMBULANCE_CAPACITY, seed=seed)

        if mode == "auto":
            self._run_auto(data_parser, scorekeeper, log)
            return

        # All GUI modes share the same UI constructor — factory provides components
        classifier, recommender, auto_execute, show_classifier_label = _make_components(
            mode, self.data_dir
        )

        UI(
            data_parser,
            scorekeeper,
            self.data_dir,
            log=log,
            classifier=classifier,
            recommender=recommender,
            auto_execute=auto_execute,
            show_classifier_label=show_classifier_label,
        )

    def _run_auto(
        self,
        data_parser: DataParser,
        scorekeeper: ScoreKeeper,
        log: bool,
    ) -> dict:
        """Run the headless auto loop: RL agent plays through the full dataset.

        Args:
            data_parser: DataParser supplying humanoids.
            scorekeeper: ScoreKeeper tracking game state.
            log: If True, write a ``log.csv`` when the loop ends.

        Returns:
            Score dict from :meth:`~gameplay.scorekeeper.ScoreKeeper.get_score`
            with keys ``killed``, ``saved``, and ``saved_value``.
        """
        classifier = CNNClassifier(img_data_root=self.data_dir)
        recommender = RLRecommender()

        # while/else: the else block runs only when the loop exits without break
        # (i.e. pool exhausted). A break (time expired) skips the else clause.
        while len(data_parser.unvisited) > 0:
            if scorekeeper.remaining_time <= 0:
                print("Ran out of time")
                break
            humanoid = data_parser.get_random()
            probs = classifier.classify(humanoid)
            action = recommender.recommend(humanoid, probs, scorekeeper)
            predicted_class = int(np.argmax(probs))
            max_prob = float(np.max(probs))
            confidence = (
                "high"
                if max_prob >= CNN_CONFIDENCE_HIGH
                else "mid" if max_prob >= CNN_CONFIDENCE_MID else "low"
            )
            action_executed = scorekeeper.map_do_action(action, humanoid)
            if action_executed:
                scorekeeper.log_turn(
                    humanoid,
                    action.value,
                    predicted_class=predicted_class,
                    recommended_action=action.value,
                    confidence=confidence,
                    obs=None,
                )
            else:
                print(
                    "Warning, no action executed in run_auto loop — this should not happen. Check map_do_action for details."
                )
        else:
            print("The City Is Empty")

        score = scorekeeper.get_score()  # scrams remaining passengers, finalizes score
        if log:
            scorekeeper.save_log()
        print("RL equiv reward:", scorekeeper.get_cumulative_reward())
        print(score)
        return score


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Extracted from ``__main__`` so that tests can import and exercise the
    parser without launching the game or training loop.

    Returns:
        Configured :class:`argparse.ArgumentParser` for the SGAI entry point.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="SGAI game — run with -m to choose a mode",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="human",
        choices=["human", "machine", "human_cnn", "human_rl", "human_cnn_rl", "auto"],
        help=(
            "Game mode (default: %(default)s). "
            "human=pure human play; "
            "machine=RL plays step-by-step with Advance Turn button; "
            "human_cnn=CNN classifies, human decides; "
            "human_rl=RL recommends action, human decides (CNN runs internally, label hidden); "
            "human_cnn_rl=CNN+RL both assist, human decides; "
            "auto=fully headless RL play (no GUI)"
        ),
    )
    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Write a log.csv after the game ends (default: off)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=DATA_DIR,
        help="Path to data directory (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility — omit to disable seeding. "
        "Useful with -m auto.",
    )

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model (RL or CNN). This is a utility, not a game mode.",
    )
    train_parser.add_argument(
        "--type",
        type=str,
        choices=["rl", "cnn"],
        required=True,
        help="Which model to train: rl=DQN agent, cnn=image classifier",
    )
    train_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=None,
        help=f"Training iterations (episodes for RL, epochs for CNN). Default: {NUM_EPISODES} for RL, {NUM_EPOCHS} for CNN.",
    )
    train_parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=DATA_DIR,
        help="Path to data directory (default: %(default)s)",
    )
    train_parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed — omit to disable seeding",
    )
    train_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save trained weights. Default: each script's own default path.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _seed_all(args.seed)
        if args.type == "rl":
            train_rl(
                data_dir=args.data,
                episodes=args.epochs if args.epochs is not None else NUM_EPISODES,
                seed=args.seed,
                save_path=args.output if args.output is not None else RL_MODEL_PATH,
            )
        else:
            train_cnn(
                data_dir=args.data,
                epochs=args.epochs if args.epochs is not None else NUM_EPOCHS,
                seed=args.seed,
                save_path=(
                    args.output if args.output is not None else CLASSIFIER_MODEL_PATH
                ),
            )
    else:
        Main(args.mode, args.log, args.data, args.seed)
