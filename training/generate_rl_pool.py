"""Generate the game image pool FROM the trained RL red-team agent.

Each zombie is disguised by the agent itself: we run its greedy policy at malice budget
2 / 8 / 10 -> low / med / high tiers (the study's independent variable). The disguise is
two-phase — the agent first fully recolors the green skin (fools the human), then tilts
the scene to walk the CNN toward its decision boundary (makes the AI unsure). As the
budget rises the AI assistant shifts from "helps" (low budget, CNN a confident zombie)
to "unsure" (top budget, CNN ~57% torn) — an emergent answer to the research question.

The agent decides its own edits and is allowed to STOP: it tilts only until one more
step would overshoot the boundary into a confidently-wrong verdict, then stops. So a
higher budget can only match or improve the disguise, never make it worse. Humans are
emitted unedited AND tilted, so "tilted" never becomes a free shortcut for "zombie"
(rotation must appear on both classes).

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

# Malice tiers ARE the agent's budget: we run the trained agent at each budget and take
# the disguise it settles on. malice_tier in {0, 2, 8, 10}; tier 0 = an unedited real
# human (the control). Budget is the study's independent variable — the more malice the
# agent is ALLOWED, the further it disguises: at budget 2 it only starts recoloring (the
# CNN is still a confident zombie), by budget 8 the human is fooled and the CNN is
# wavering, and at budget 10 the CNN is walked to its decision boundary (~57% of figures
# uncertain). The agent decides its own edits and STOPS when more would only overshoot,
# so a higher budget never forces a worse disguise.
BUDGET_TIERS = (2, 8, 10)
HUMAN_TIER = 0
ROT_ANGLE = 6.0                               # fallback human tilt (deg) if no zombie
                                              # tilts exist to sample from; see below
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


def _setup_env_on(env, path, budget):
    """Point the env at a specific zombie image with a specific starting budget."""
    im = Image.open(os.path.join(env.img_data_root, path)).convert("RGB")
    env.original_image = im.copy()
    env.green_mask = green_mask(env.original_image)
    env.cum_angle = 0.0
    env.cum_recolor = 0.0
    env.budget_left = budget
    env.prev_score = env._score_and_observe()
    return env._get_observation()


def _rollout_at_budgets(env, agent, path):
    """Run the greedy agent on one zombie at each BUDGET_TIER, returning {budget: state}.

    Each tier is an INDEPENDENT rollout at that budget — the malice budget IS the tier.
    STOP is allowed (not masked): the agent tilts the CNN toward its boundary and stops
    when a further edit would only overshoot it, so we take exactly the disguise the agent
    judges best for that budget. That is why raising the budget can only match or improve
    the disguise, never make it confidently wrong.
    """
    out = {}
    for budget in BUDGET_TIERS:
        obs = _setup_env_on(env, path, budget)
        while True:
            action = agent.select_action(obs, greedy=True, available=env.action_mask())
            if action == env.STOP:
                break
            obs, _r, _term, truncated, _info = env.step(action)
            if truncated:              # budget spent
                break
        out[budget] = (env.current_image.copy(), env.cum_recolor, env.cum_angle,
                       env.last_hue, float(env.last_probs[1]))
    return out


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
            "malice_tier": malice_tier,         # 0 = real human control; 2/8/10 = agent budget
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

    # --- Zombies: RL agent disguises each at budget 2 / 8 / 10 (the tiers) ---
    for _, row in df[df["Class"] == "Zombie"].iterrows():
        src = row["Filename"]
        mask = green_mask(Image.open(os.path.join(data_dir, src)).convert("RGB"))
        stem = Path(src).stem
        tiers = _rollout_at_budgets(env, agent, src)
        for budget in BUDGET_TIERS:
            img, recolor, angle, _hue, _ph = tiers[budget]
            # the tier IS the malice budget: 2 / 8 / 10
            record(img, f"{stem}_tier{budget}.png", "zombie", budget, recolor, angle,
                   source=stem, mask=mask)

    # --- Humans: unedited + tilted control (rotation on both classes) ---
    # Sample the human tilt MAGNITUDE from the actual zombie tilt distribution, so that
    # "how much it's tilted" can never separate the classes — only WHETHER it's tilted,
    # which the curator matches per session. The agent tilts zombies only ~0.5-3 deg; a
    # fixed larger angle here would make "heavily tilted = human", a brand-new shortcut.
    zombie_tilts = [abs(r["angle"]) for r in rows
                    if r["true_class"] == "zombie" and abs(r["angle"]) > 0.01]
    for _, row in df[df["Class"] == "Default"].iterrows():
        src = row["Filename"]
        base = Image.open(os.path.join(data_dir, src)).convert("RGB")
        stem = Path(src).stem
        record(base, f"{stem}_tier0.png", "human", HUMAN_TIER, 0.0, 0.0, source=stem)
        mag = rng.choice(zombie_tilts) if zombie_tilts else ROT_ANGLE
        angle = mag if rng.random() < 0.5 else -mag
        tilted = TRANSFORM_REGISTRY["rotation"].apply(base, angle / 30.0 + 0.5)
        record(tilted, f"{stem}_tier0_tilt.png", "human", HUMAN_TIER, 0.0, round(angle, 1),
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
    ).reindex(list(BUDGET_TIERS)).round(3)
    print(summary)
    print("\n(hue_proxy up = human fooled more; assistant_correct down = AI fooled more)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default="data/rl_pool")
    ap.add_argument("--agent", default="models/weights/adversarial_agent.pth")
    args = ap.parse_args()
    generate(args.data, args.out, args.agent)
