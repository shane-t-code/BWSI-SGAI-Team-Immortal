"""Generate the game image pool FROM the trained RL red-team agent.

Each zombie is disguised by the agent itself: we run its greedy policy and snapshot
the image after 1 / 3 / 6 edits -> low / med / high malice-budget tiers (the study's
independent variable). The agent chooses its own mix of recolor (fools the human) and
rotation (fools the CNN), so as budget rises the disguise strengthens on both channels
and the AI assistant naturally shifts from "helps" (low budget, CNN still right) to
"misleads" (high budget, CNN fooled) — an emergent answer to the research question.

"Stop" is disabled during generation so every zombie spends the full budget and yields
a clean 1/3/6 snapshot. Humans are emitted unedited AND tilted, so "tilted" never
becomes a free shortcut for "zombie" (rotation must appear on both classes).

Run:
    python training/generate_rl_pool.py                 # -> data/rl_pool/
    python training/generate_rl_pool.py --agent models/weights/adversarial_agent.pth
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from gameplay.constants import DATA_DIR  # noqa: E402
from models.CNNClassifier import CNNClassifier  # noqa: E402
from models.DQNAgent import DQNAgent  # noqa: E402
from models.image_transforms import TRANSFORM_REGISTRY, green_mask, hue_proxy  # noqa: E402
from game_interfaces.adversarial_env import AdversarialEnv  # noqa: E402

# Malice tiers ARE the edit counts: we snapshot the agent's rollout after this many
# edits. Matches docs/data-logging-schema.md, where malice_tier in {0, 1, 3, 6} and
# tier 0 = an unedited real human (the control).
SNAPSHOT_EDITS = (1, 3, 6)
HUMAN_TIER = 0
ROT_ANGLE = 12.0                              # tilt magnitude for human controls
CLASS_NAMES = ["zombie", "healthy", "injured", "corpse"]


def _verdict(im, clf):
    """What the CNN (the AI assistant) would report about this image.

    Returns (p_healthy, max_prob, margin, predicted_class_name, says_human):
        p_healthy: P(healthy) — the raw disguise-success number, for our analysis.
        max_prob:  confidence in whatever it predicted. THIS is what the UI shows —
                   not P(healthy), which would read as "0.5% confident" on an image
                   the CNN is 99.5% sure is a zombie.
        margin:    gap between its top two classes; small = the CNN is torn.
        says_human: True if the top class is a living-human class (healthy/injured).
    """
    p = clf.classify_pil_image(im)
    order = np.argsort(p)[::-1]
    pred = int(order[0])
    margin = float(p[order[0]] - p[order[1]])
    return float(p[1]), float(p[pred]), margin, CLASS_NAMES[pred], pred in (1, 2)


def _setup_env_on(env, path):
    """Point the env at a specific zombie image (bypasses random reset)."""
    im = Image.open(os.path.join(env.img_data_root, path)).convert("RGB")
    env.original_image = im.copy()
    env.green_mask = green_mask(env.original_image)
    env.cum_angle = 0.0
    env.cum_recolor = 0.0
    env.budget_left = env.max_budget
    env.prev_score = env._score_and_observe()
    return env._get_observation()


def _rollout_snapshots(env, agent, path):
    """Run the greedy agent on one zombie, returning {edits: full_image} at each snapshot."""
    obs = _setup_env_on(env, path)
    snaps = {}
    step = 0
    done = False
    while not done:
        mask = env.action_mask()
        if any(mask[: env.STOP]):      # if any edit is legal, forbid stopping early
            mask[env.STOP] = False
        action = agent.select_action(obs, greedy=True, available=mask)
        obs, _r, _term, truncated, _info = env.step(action)
        step += 1
        done = truncated               # ends when budget runs out (stop is masked off)
        if step in SNAPSHOT_EDITS:
            snaps[step] = (env.current_image.copy(), env.cum_recolor, env.cum_angle,
                           env.last_hue, float(env.last_probs[1]))
    return snaps


def generate(data_dir, out_dir, agent_path, seed=20260715):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    clf = CNNClassifier(img_data_root=data_dir)
    env = AdversarialEnv(clf, img_data_root=data_dir)
    agent = DQNAgent(obs_size=env.observation_space.shape[0],
                     num_actions=int(env.action_space.n))
    agent.load(agent_path)

    df = pd.read_csv(os.path.join(data_dir, "consolidated_metadata.csv"))
    rows = []

    def record(img, name, true_class, malice_tier, recolor, angle, source, mask=None):
        """Save one pool image and log its facts in docs/data-logging-schema.md terms."""
        img.save(os.path.join(out_dir, name))
        p_healthy, max_prob, margin, pred, says_human = _verdict(img, clf)
        is_human = true_class == "human"
        rows.append({
            # --- fields the trial log joins on / copies (see data-logging-schema.md) ---
            "subject_id": Path(name).stem,      # the game's ID for this image
            "filename": name,
            "source": source,                   # the underlying render, for disjoint sessions
            "malice_tier": malice_tier,         # 0 = real human control; 1/3/6 = edit counts
            "ground_truth_human": is_human,
            # confidence in the CNN's OWN verdict (0-100) — this is what the UI displays.
            "ai_confidence_shown": round(max_prob * 100.0, 2),
            "cnn_says_human": says_human,       # what the assistant would tell the player
            "cnn_correct": says_human == is_human,
            "cnn_predicted": pred,              # drives the threat-advisory text
            "cnn_margin": round(margin, 4),     # small = "Ambiguous Classification"
            # --- extra diagnostics (ours, not part of the trial schema) ---
            "true_class": true_class,
            "cnn_p_healthy": round(p_healthy, 4),
            "recolor": round(recolor, 2),
            "angle": round(angle, 1),
            "hue_proxy": round(hue_proxy(img, mask), 4) if mask is not None else 1.0,
        })

    # --- Zombies: RL agent disguises each, snapshotted at 1/3/6 edits ---
    for _, row in df[df["Class"] == "Zombie"].iterrows():
        src = row["Filename"]
        mask = green_mask(Image.open(os.path.join(data_dir, src)).convert("RGB"))
        stem = Path(src).stem
        snaps = _rollout_snapshots(env, agent, src)
        for edits in SNAPSHOT_EDITS:
            if edits not in snaps:
                continue
            img, recolor, angle, _hue, _ph = snaps[edits]
            # the tier IS the edit count: 1 / 3 / 6
            record(img, f"{stem}_tier{edits}.png", "zombie", edits, recolor, angle,
                   source=stem, mask=mask)

    # --- Humans: unedited + tilted control (rotation on both classes) ---
    for _, row in df[df["Class"] == "Default"].iterrows():
        src = row["Filename"]
        base = Image.open(os.path.join(data_dir, src)).convert("RGB")
        stem = Path(src).stem
        record(base, f"{stem}_tier0.png", "human", HUMAN_TIER, 0.0, 0.0, source=stem)
        angle = ROT_ANGLE if rng.random() < 0.5 else -ROT_ANGLE
        tilted = TRANSFORM_REGISTRY["rotation"].apply(base, angle / 30.0 + 0.5)
        record(tilted, f"{stem}_tier0_tilt.png", "human", HUMAN_TIER, 0.0, angle,
               source=stem)

    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(out_dir, "pool_metadata.csv"), index=False)

    # --- validation: disguise should strengthen with budget on BOTH channels ---
    z = meta[meta.true_class == "zombie"]
    print(f"wrote {len(meta)} images to {out_dir}\n")
    print("ZOMBIES by malice_tier (disguise should rise with the budget):")
    summary = z.groupby("malice_tier").agg(
        n=("filename", "size"),
        mean_recolor=("recolor", "mean"),
        mean_hue_human=("hue_proxy", "mean"),
        mean_cnn_p_healthy=("cnn_p_healthy", "mean"),
        assistant_correct=("cnn_correct", "mean"),
    ).reindex(list(SNAPSHOT_EDITS)).round(3)
    print(summary)
    print("\n(hue_proxy up = human fooled more; assistant_correct down = AI fooled more)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default="data/rl_pool")
    ap.add_argument("--agent", default="models/weights/adversarial_agent.pth")
    args = ap.parse_args()
    generate(args.data, args.out, args.agent)
