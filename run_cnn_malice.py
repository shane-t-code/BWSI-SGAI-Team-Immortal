"""Run CNN classifier on malice budget (adversarial) images."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from models.CNNClassifier import CNNClassifier
from gameplay.humanoid import Humanoid


def main():
    """Run CNN on all malice budget images and display results."""
    
    # Initialize CNN classifier
    print("Loading CNN classifier...")
    classifier = CNNClassifier(img_data_root="data")
    print("CNN loaded successfully!\n")
    
    # Load game_manifest.csv to get malice_tier mapping
    manifest_path = Path("data/game_sets/game_manifest.csv")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        tier_map = dict(zip(manifest["filename"], manifest["malice_tier"]))
        print(f"Loaded tier mapping from game_manifest.csv ({len(tier_map)} entries)\n")
    else:
        print("Warning: game_manifest.csv not found, will parse tiers from filenames")
        tier_map = {}
    
    # Get all malice budget images
    malice_dir = Path("data/game_sets/images")
    if not malice_dir.exists():
        print(f"Error: Directory {malice_dir} not found")
        return
    
    # Get all PNG files
    image_files = sorted(malice_dir.glob("*.png"))
    print(f"Found {len(image_files)} malice budget images\n")
    
    # Classify each image
    results = []
    
    for img_path in image_files:
        # Create a temporary Humanoid object
        temp_humanoid = Humanoid(
            fp=str(img_path.relative_to("data")),
            state="Default",  # placeholder
            value=0.0         # placeholder
        )
        
        # Get classification
        probs = classifier.classify(temp_humanoid)
        
        # Get tier from game_manifest.csv if available, otherwise parse from filename
        filename = img_path.name
        if filename in tier_map:
            tier = tier_map[filename]
        else:
            # Fallback: parse tier level from filename (old tier set)
            if "tier10" in filename:
                tier = 10
            elif "tier8" in filename:
                tier = 8
            elif "tier5" in filename:
                tier = 5
            elif "tier2" in filename:
                tier = 2
            elif "tier1" in filename:
                tier = 1
            elif "tier0" in filename:
                tier = 0
            else:
                tier = "unknown"
        
        # Get predicted class and confidence
        predicted_class_idx = int(np.argmax(probs))
        class_names = ["zombie", "healthy", "injured", "corpse"]
        predicted_class = class_names[predicted_class_idx]
        confidence = float(np.max(probs))
        
        results.append({
            "filename": filename,
            "tier": tier,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "zombie_prob": float(probs[0]),
            "healthy_prob": float(probs[1]),
            "injured_prob": float(probs[2]),
            "corpse_prob": float(probs[3])
        })
    
    # Display results
    print("=" * 100)
    print("CNN CLASSIFICATION RESULTS ON MALICE BUDGET IMAGES")
    print("=" * 100)
    print(f"{'Filename':<40} {'Tier':<6} {'Predicted':<10} {'Confidence':<10} {'Zombie':<8} {'Healthy':<8} {'Injured':<8} {'Corpse':<8}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['filename']:<40} {str(result['tier']):<6} {result['predicted_class']:<10} {result['confidence']:.4f}    {result['zombie_prob']:.4f}  {result['healthy_prob']:.4f}  {result['injured_prob']:.4f}  {result['corpse_prob']:.4f}")
    
    print("=" * 100)
    
    # Summary statistics by tier
    print("\nSUMMARY BY TIER LEVEL:")
    print("=" * 60)
    
    tier_stats = {}
    for result in results:
        tier = result['tier']
        if tier not in tier_stats:
            tier_stats[tier] = {
                'count': 0,
                'avg_confidence': 0.0,
                'zombie_predictions': 0,
                'healthy_predictions': 0,
                'injured_predictions': 0,
                'corpse_predictions': 0
            }
        tier_stats[tier]['count'] += 1
        tier_stats[tier]['avg_confidence'] += result['confidence']
        
        if result['predicted_class'] == 'zombie':
            tier_stats[tier]['zombie_predictions'] += 1
        elif result['predicted_class'] == 'healthy':
            tier_stats[tier]['healthy_predictions'] += 1
        elif result['predicted_class'] == 'injured':
            tier_stats[tier]['injured_predictions'] += 1
        elif result['predicted_class'] == 'corpse':
            tier_stats[tier]['corpse_predictions'] += 1
    
    for tier in sorted(tier_stats.keys()):
        stats = tier_stats[tier]
        avg_conf = stats['avg_confidence'] / stats['count'] if stats['count'] > 0 else 0
        print(f"Tier {tier}: {stats['count']} images | Avg Confidence: {avg_conf:.4f}")
        print(f"  Predictions: Zombie={stats['zombie_predictions']}, Healthy={stats['healthy_predictions']}, Injured={stats['injured_predictions']}, Corpse={stats['corpse_predictions']}")
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_path = "malice_budget_cnn_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
