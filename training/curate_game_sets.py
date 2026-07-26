"""Curate balanced, tilt-matched, disjoint game sets from the RL pool.

The raw pool (training/generate_rl_pool.py) is a catalog: every zombie at every tier,
every human in both forms. This turns it into game-ready sessions, closing three
threats to the experiment by construction:

  1. BASE RATE. Tier 0 IS the humans and tiers 2/8/10 are the zombies, so equal trials
     per tier would be 25% human -- a player who always says "zombie" scores 75%
     without looking. We emit 50/50, which makes always-guessing worth exactly nothing.

  2. TILT LEAK. The agent tilts more as budget rises (0% of tier-2 zombies are tilted,
     ~95% of tier-8 and tier-10). If humans aren't tilted at a matching rate AND
     magnitude, "tilted = zombie" is free information. We match the human tilt rate to
     the measured zombie tilt rate per session, and human tilt ANGLES are sampled from
     the zombie tilt distribution (generate_rl_pool.py), so with 50/50 classes:
         P(zombie | tilted) == P(zombie | upright) == 0.5
     i.e. tilt carries ZERO information. Not reduced -- eliminated.

  3. REPEAT PLAYERS. People will play more than once. Sessions are disjoint at the
     SOURCE level (not the file level): each underlying render appears in at most one
     session, in one form. File-level disjointness isn't enough -- test_00004_tier1
     and test_00004_tier6 are different files but the SAME zombie on the same street,
     and a replayer recognizes the scene instead of judging it.

Threat-advisory text is derived ONLY from what the CNN actually output (its predicted
class, its confidence, how torn it is). The CNN reports four probabilities -- it does
not detect "asymmetrical features" or "unnatural gait", so we never claim it did.
Inventing the AI's reasoning would mean studying trust in fiction we wrote.

Run:
    python training/curate_game_sets.py --pool data/rl_pool
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

HUMAN_TIER = 0   # tier 0 = unedited real human (the control)
# NOTE: the zombie malice tiers are read from the pool metadata at runtime, never
# hardcoded here — generate_rl_pool.py owns them, and duplicating the list in two
# files is exactly how they silently drift apart.

# CNN predicted class -> advisory line. One line per thing the CNN can actually say.
THREAT_TEXT = {
    "healthy": "No Anomalies Detected",
    "injured": "Trauma Indicators Present",
    "zombie": "Overgrowth Signature Detected",
    "corpse": "No Vital Signs Detected",
}
AMBIGUOUS_MARGIN = 0.25   # top-2 gap below this -> the CNN is genuinely torn
HIGH_CONFIDENCE = 70.0    # ai_confidence_shown at/above this reads as "high"


def derive_threats(row) -> list[str]:
    """Build the THREAT ADVISORY lines from the CNN's real output for this image.

    Every line corresponds to something the classifier actually produced: its verdict,
    whether it was torn, and how confident it was. Nothing here is invented.

    Note this is what makes the panel honest rather than an oracle: at tier 6 the CNN
    is fooled, so a disguised zombie reads "No Anomalies Detected / High AI Confidence"
    -- confidently wrong, exactly as unreliable as the classifier itself.
    """
    lines = [THREAT_TEXT[row.cnn_predicted]]
    if row.cnn_margin < AMBIGUOUS_MARGIN:
        lines.append("Ambiguous Classification")
    lines.append("High AI Confidence" if row.ai_confidence_shown >= HIGH_CONFIDENCE
                 else "Low AI Confidence")
    return lines


def load_injury_lookup(metadata_path: str) -> dict:
    """Map each source stem (e.g. "test_00190") to its Injured flag.

    consolidated_metadata.csv is keyed by the original render, not the
    per-tier pool image, so this is a separate lookup rather than a column
    generate_rl_pool.py could just carry through: `source` in the pool/game
    manifest already IS that same stem (Path(Filename).stem), which is what
    makes the join exact.
    """
    df = pd.read_csv(metadata_path)
    stems = df["Filename"].map(lambda f: Path(f).stem)
    return dict(zip(stems, df["Injured"]))


def build_sessions(meta: pd.DataFrame, per_tier: int, n_sessions: int, seed: int):
    """Assemble disjoint, class-balanced, tilt-matched sessions.

    Args:
        meta: the pool metadata (one row per pool image).
        per_tier: zombies per zombie-tier per session (5 -> 30-round sessions).
        n_sessions: how many disjoint sessions to build.
        seed: for reproducible sampling.

    Returns:
        A DataFrame of selected rounds with a `session` column.
    """
    rng = random.Random(seed)
    zombies = meta[~meta.ground_truth_human]
    humans = meta[meta.ground_truth_human]

    # Read the malice tiers straight from the pool — whatever generate_rl_pool.py
    # produced is the truth, so changing tiers there can never break this script.
    zombie_tiers = sorted(int(t) for t in zombies.malice_tier.unique())

    zombie_sources = sorted(zombies.source.unique())
    human_sources = sorted(humans.source.unique())
    rng.shuffle(zombie_sources)
    rng.shuffle(human_sources)

    per_session = per_tier * len(zombie_tiers)          # zombies per session
    need = n_sessions * per_session                      # ... and the same many humans
    if need > len(zombie_sources):
        raise SystemExit(
            f"Need {need} distinct zombie sources for {n_sessions} disjoint sessions "
            f"at --per-tier {per_tier}, but the pool has {len(zombie_sources)}. "
            f"Lower --sessions or --per-tier (max sessions here: "
            f"{len(zombie_sources) // per_session})."
        )
    if need > len(human_sources):
        raise SystemExit(
            f"Need {need} distinct human sources but the pool has {len(human_sources)}."
        )

    picked, zi, hi = [], 0, 0
    for session in range(1, n_sessions + 1):
        # --- zombies: per_tier distinct sources at each malice tier ---
        session_rows = []
        for tier in zombie_tiers:
            for _ in range(per_tier):
                src = zombie_sources[zi]; zi += 1
                row = zombies[(zombies.source == src)
                              & (zombies.malice_tier == tier)].iloc[0]
                session_rows.append(row)

        # --- measure the ACTUAL tilt rate of the zombies we just picked ---
        # (measured, not assumed: a 5-image sample won't land exactly on 30%/44%/100%)
        n_tilted = sum(1 for r in session_rows if r.angle != 0)

        # --- humans: same count, same tilt rate -> tilt becomes uninformative ---
        for i in range(per_session):
            src = human_sources[hi]; hi += 1
            want_tilted = i < n_tilted
            cand = humans[(humans.source == src)
                          & ((humans.angle != 0) == want_tilted)]
            session_rows.append(cand.iloc[0])

        for r in session_rows:
            picked.append({**r.to_dict(), "session": session})

    out = pd.DataFrame(picked)
    out["threats"] = [derive_threats(r) for r in out.itertuples()]
    # Ground truth is always the subject's real class -- a wounded-but-alive
    # human is still PERMIT, even at tier 0 where the CNN itself misreads them
    # as a corpse ("No Vital Signs Detected"). Scoring off cnn_predicted would
    # make ground truth depend on the classifier's own error, and would grade
    # the CNN using the CNN's output. That confusability is real and worth
    # keeping visible -- see subject_injured below -- it just isn't a
    # different correct answer.
    out["ground_truth"] = out["ground_truth_human"].map({True: "human", False: "zombie"})
    return out


def main(pool_dir: str, out_dir: str, per_tier: int, n_sessions: int, seed: int,
         metadata_path: str) -> None:
    meta = pd.read_csv(os.path.join(pool_dir, "pool_metadata.csv"))
    os.makedirs(out_dir, exist_ok=True)
    sets = build_sessions(meta, per_tier, n_sessions, seed)

    # subject_injured is descriptive, not scoring: whether a human subject is
    # wounded doesn't change the correct PERMIT/REJECT answer (see
    # build_sessions), but it does let analysis ask "do players reject
    # injured humans more often?" -- especially at tier 0, where the CNN's
    # own "No Vital Signs Detected" misread on injured subjects is the
    # clearest over-trust case in the dataset.
    # Gated on ground_truth_human: consolidated_metadata.csv carries an Injured
    # flag for EVERY source render, zombies included, so the raw lookup marks
    # wounded-looking zombies as injured too. "Injured" here means specifically
    # a wounded-but-alive human (a PERMIT the player might wrongly turn away);
    # an overgrown is a REJECT whether or not its render looks wounded. Without
    # this gate the flag is true for far more zombies than humans, and any
    # "do players reject injured humans more often?" analysis run on it would
    # be dominated by zombies rather than measuring the intended effect.
    injury_lookup = load_injury_lookup(metadata_path)
    sets["subject_injured"] = (
        sets["source"].map(injury_lookup).fillna(False).astype(bool)
        & sets["ground_truth_human"].astype(bool)
    )

    # --- hard guarantee: no source may appear in two sessions (repeat-player safety) ---
    per_source = sets.groupby("source")["session"].nunique()
    assert per_source.max() == 1, (
        f"source reuse across sessions: {per_source[per_source > 1].index.tolist()[:5]}"
    )

    # Copy the chosen images in, so this folder is self-contained and the game needs
    # nothing else. The raw pool (~81MB, 490 images) stays a regenerable intermediate
    # and is NOT committed — it rebuilds from the agent weights + generate_rl_pool.py.
    # Wipe the folder first: if the tiers change (e.g. 1/5/10 -> 2/8/10), the old tier
    # images would otherwise linger here as orphans, unreferenced but committed.
    # Clear the directory's contents rather than rmtree-ing the directory itself:
    # on OneDrive-synced checkouts this folder is a Files-on-Demand reparse point,
    # and OneDrive's filter driver denies the final rmdir() even after every file
    # inside has been removed.
    img_dir = os.path.join(out_dir, "images")
    if os.path.isdir(img_dir):
        for entry in os.scandir(img_dir):
            if entry.is_dir():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
    os.makedirs(img_dir, exist_ok=True)
    for fname in sets.filename:
        shutil.copy2(os.path.join(pool_dir, fname), os.path.join(img_dir, fname))

    # ground_truth_human is the raw true-class boolean used to build the sessions
    # above; the manifest exports only the single "human"/"zombie" label
    # (ground_truth) that the game actually scores against.
    sets.drop(columns=["ground_truth_human"]).to_csv(
        os.path.join(out_dir, "game_manifest.csv"), index=False
    )

    # --- rounds.json: shaped like Claire's ROUNDS array, one array per session ---
    rounds = {}
    for session, grp in sets.groupby("session"):
        rounds[str(session)] = [
            {
                "subjectId": r.subject_id,
                "image": r.filename,
                "tier": int(r.malice_tier),
                "groundTruth": r.ground_truth,
                # subject_injured is NaN for any source missing from the metadata
                # lookup; NaN is truthy in Python, so this must be checked
                # explicitly rather than passed through bool().
                "isInjured": bool(r.subject_injured) if pd.notna(r.subject_injured) else False,
                "aiConfidence": float(r.ai_confidence_shown),
                # What the CNN actually recommended (human vs zombie). The game logs this
                # per trial so "did the player follow the AI" / over-trust is computable
                # without joining subject_id back to the manifest. aiConfidence alone is
                # directionless (it's confidence in whatever class it picked, not which one).
                "cnnSaysHuman": bool(r.cnn_says_human),
                "threats": r.threats,
            }
            for r in grp.itertuples()
        ]
    with open(os.path.join(out_dir, "rounds.json"), "w") as f:
        json.dump(rounds, f, indent=2)

    # ---------------- validation ----------------
    print(f"{len(sets)} rounds across {n_sessions} disjoint sessions -> {out_dir}\n")
    print("per session (tilt rates must match, or 'tilted' leaks 'zombie'):")
    for session, g in sets.groupby("session"):
        z, h = g[g.ground_truth == "zombie"], g[g.ground_truth != "zombie"]
        zt = (z.angle != 0).mean()
        ht = (h.angle != 0).mean()
        print(f"  session {session}: {len(g)} rounds | {len(z)} zombie / {len(h)} human "
              f"({100*len(h)/len(g):.0f}% human) | tilt: zombie {zt:.0%} vs human {ht:.0%}"
              f"{'  <-- MISMATCH' if abs(zt - ht) > 0.01 else ''}")
    print("\nsource disjointness across sessions: OK (asserted)")
    n_injured = int(sets["subject_injured"].fillna(False).sum())
    n_injured_misread = int((sets["subject_injured"].fillna(False)
                              & (sets["cnn_predicted"] == "corpse")).sum())
    print(f"subject_injured: {n_injured} injured humans in the curated sessions "
          f"({n_injured_misread} of those the CNN itself reads as a corpse -- "
          f"still ground_truth 'human', still PERMIT)")
    print("\nassistant reliability by tier (what SCAN/AI panel will tell players):")
    print(sets.groupby("malice_tier")["cnn_correct"].agg(["size", "mean"]).round(3))
    print("\nsample threat lines:")
    for tier in sorted(sets.malice_tier.unique()):
        r = sets[sets.malice_tier == tier].iloc[0]
        print(f"  tier {tier} ({r.ground_truth.upper():6s}): {r.threats}  [conf {r.ai_confidence_shown}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/rl_pool", help="dir with pool_metadata.csv")
    ap.add_argument("--out", default="data/game_sets")
    ap.add_argument("--per-tier", type=int, default=5, help="zombies per tier per session")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--metadata", default="data/consolidated_metadata.csv",
                     help="path to consolidated_metadata.csv, for subject_injured")
    args = ap.parse_args()
    main(args.pool, args.out, args.per_tier, args.sessions, args.seed, args.metadata)
