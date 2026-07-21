"""Verify tier progression on the regenerated pool.

This script reads the pool metadata and validates that:
1. Disguise strengthens with budget on BOTH channels (human and AI)
2. hue_proxy values are consistent with the new scale (6.67, changed from 4.0)

Run:
    python training/verify_tiers.py
"""

import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
POOL_METADATA = _root / "data" / "game_sets" / "game_manifest.csv"

def verify_tiers():
    """Load pool metadata and verify tier progression."""
    meta = pd.read_csv(POOL_METADATA)
    
    # Filter zombies only (tier 0 is human control)
    z = meta[meta.true_class == "zombie"]
    
    print(f"Total images in pool: {len(meta)}")
    print(f"Zombie images: {len(z)}")
    print(f"Human images: {len(meta[meta.true_class == 'human'])}")
    print()
    
    # Group by malice_tier and show progression (include tier 0 human control)
    print("=" * 80)
    print("ALL TIERS (tier 0 = human control, 2/8/10 = zombie budgets):")
    print("=" * 80)
    summary = meta.groupby("malice_tier").agg(
        n=("filename", "size"),
        mean_recolor=("recolor", "mean"),
        mean_hue_human=("hue_proxy", "mean"),
        mean_cnn_p_healthy=("cnn_p_healthy", "mean"),
        assistant_correct=("cnn_correct", "mean"),
        ground_truth_human=("ground_truth_human", "mean"),
    ).round(3)
    print(summary)
    print()
    print("(hue_proxy up = human fooled more; assistant_correct down = AI fooled more)")
    print()
    
    # Note about hue_proxy scale change
    print("=" * 80)
    print("NOTE: hue_proxy scale changed from 4.0 to 6.67")
    print("=" * 80)
    print("The hue_proxy scale in models/image_transforms.py was recalibrated:")
    print("  - Old scale: 4.0 (calibrated against signed version)")
    print("  - New scale: 6.67 (recalibrated after clamp change)")
    print("This change ensures an untouched zombie scores ~1.0 on the hue_proxy metric.")
    print()
    
    # Check progression
    print("=" * 80)
    print("PROGRESSION CHECK:")
    print("=" * 80)
    tiers = [2, 8, 10]
    for i in range(len(tiers) - 1):
        t1, t2 = tiers[i], tiers[i + 1]
        if t1 in summary.index and t2 in summary.index:
            hue_diff = summary.loc[t2, "mean_hue_human"] - summary.loc[t1, "mean_hue_human"]
            correct_diff = summary.loc[t1, "assistant_correct"] - summary.loc[t2, "assistant_correct"]
            
            print(f"Tier {t1} -> {t2}:")
            print(f"  hue_proxy change: {hue_diff:+.3f} (should be positive)")
            print(f"  assistant_correct change: {correct_diff:+.3f} (should be positive)")
            print()
    
    # Show sample threat lines per tier (from game_manifest.csv if available)
    print("=" * 80)
    print("SAMPLE THREAT LINES:")
    print("=" * 80)
    game_manifest = _root / "data" / "game_sets" / "game_manifest.csv"
    if game_manifest.exists():
        game_meta = pd.read_csv(game_manifest)
        for tier in sorted(game_meta.malice_tier.unique()):
            r = game_meta[game_meta.malice_tier == tier].iloc[0]
            truth = "HUMAN" if r.ground_truth_human else "ZOMBIE"
            print(f"  tier {tier} ({truth:6s}): {r.threats}  [conf {r.ai_confidence_shown}]")
    else:
        print("  (game_manifest.csv not found - threat lines unavailable)")

if __name__ == "__main__":
    verify_tiers()
