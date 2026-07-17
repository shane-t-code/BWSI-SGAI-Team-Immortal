"""Channel-collapse detector for adversarial DQN training.

The two-term reward (W_HUMAN * hue_proxy + W_AI * P(healthy)) only produces a real
allocation problem if the agent actually USES both attack channels. The failure mode
is "channel collapse": the agent pours its whole malice budget into one channel and
ignores the other, silently reducing the study to a single-channel attack.

Two things make this measurement honest:
  * We track BUDGET SHARE per channel, not raw action counts — a "stop" action costs
    no budget, so counting raw actions would distort the picture.
  * The report only judges collapse once epsilon has decayed. While the agent is still
    exploring it acts randomly, which spreads budget evenly across channels no matter
    what policy it has actually learned — so an early "balanced" split means nothing.

The action -> channel map is built from the ACTUAL env instance (its ROTATE_CW/RECOLOR/
... constants), so it can never drift out of sync with the env's action layout.
"""

from collections import Counter

COLLAPSE_THRESHOLD = 0.05  # a channel below 5% of spent budget is treated as dead


class ActionTracker:
    """Accumulates per-episode action counts and reports channel budget share."""

    def __init__(self, env) -> None:
        """Build the action->channel map from the env being trained.

        Args:
            env: An AdversarialEnv (or anything exposing ROTATE_CW, ROTATE_CCW,
                RECOLOR, STOP action-index constants). Reading the map from the
                env means it stays correct even if the action layout is reordered.
        """
        # Each budget-spending action belongs to one attack channel. STOP spends
        # nothing, so it maps to None and is excluded from the share math.
        self.action_channel: dict[int, str | None] = {
            env.ROTATE_CW: "rotate",    # tilt the scene   -> fools the CNN
            env.ROTATE_CCW: "rotate",   # tilt the scene   -> fools the CNN
            env.RECOLOR: "recolor",     # green -> skin    -> fools the human
            env.STOP: None,             # submit; costs no budget
        }
        # Both channels are meaningful now (orthogonal): a dead one means the agent
        # gave up on either fooling the human (recolor) or fooling the CNN (rotate).
        self.channels = ["rotate", "recolor"]
        self.meaningful = ["rotate", "recolor"]

        self.episode_counts: list[Counter] = []

    def start_episode(self) -> None:
        """Begin a fresh per-episode tally. Call once right after env.reset()."""
        self.episode_counts.append(Counter())

    def record(self, action: int) -> None:
        """Note one action the agent took this episode."""
        self.episode_counts[-1][action] += 1

    def channel_share(self, last_n: int | None = None) -> dict[str, float]:
        """Fraction of *spent budget* going to each channel over the last N episodes.

        Args:
            last_n: Only consider the most recent N episodes (None = all of them).

        Returns:
            A dict channel -> fraction in [0, 1]. Fractions sum to 1 across the
            budget-spending channels (STOP is excluded).
        """
        window = self.episode_counts[-last_n:] if last_n else self.episode_counts
        spend: Counter = Counter()
        for episode in window:
            for action, count in episode.items():
                channel = self.action_channel[action]
                if channel is not None:  # STOP costs no budget
                    spend[channel] += count

        total = sum(spend.values())
        if total == 0:
            return {ch: 0.0 for ch in self.channels}
        return {ch: spend[ch] / total for ch in self.channels}

    def report(self, last_n: int = 50, epsilon: float | None = None) -> str:
        """One-line human-readable channel-share summary with a collapse flag.

        Args:
            last_n: Window of recent episodes to summarize.
            epsilon: Current exploration rate. If still high (> 0.15) the report
                says the numbers aren't meaningful yet instead of crying collapse.

        Returns:
            A formatted string for printing alongside the training log line.
        """
        share = self.channel_share(last_n)
        bar = "  ".join(f"{ch}:{share[ch]:5.1%}" for ch in self.channels)

        # Still exploring -> a random policy spreads budget evenly regardless of
        # what was learned, so the split tells us nothing yet.
        if epsilon is not None and epsilon > 0.15:
            return f"{bar}   (exploring, eps={epsilon:.2f} -- not yet meaningful)"

        # Past exploration: flag any MEANINGFUL channel that has gone dead.
        dead = [ch for ch in self.meaningful if share[ch] < COLLAPSE_THRESHOLD]
        if dead:
            return f"{bar}   *** COLLAPSE: {', '.join(dead)} unused -> rebalance weights"
        return f"{bar}   ok -- both channels in use"


if __name__ == "__main__":
    import random

    # Tiny stub with the same action-index constants as AdversarialEnv, so this
    # demo runs without loading the CNN. Shows what a healthy vs. collapsed split
    # looks like, so the real training output is recognizable when it appears.
    class _StubEnv:
        ROTATE_CW, ROTATE_CCW, RECOLOR, STOP = 0, 1, 2, 3

    for label, policy in [
        ("balanced agent", [0, 1, 2, 2, 0, 3]),           # uses rotate + recolor
        ("collapsed onto rotation", [0, 1, 0, 1, 0, 1]),  # only ever rotates
    ]:
        tracker = ActionTracker(_StubEnv())
        for _ in range(60):
            tracker.start_episode()
            for _ in range(6):  # a 6-unit budget
                tracker.record(random.choice(policy))
        print(f"{label:26s} | {tracker.report(last_n=50, epsilon=0.05)}")
