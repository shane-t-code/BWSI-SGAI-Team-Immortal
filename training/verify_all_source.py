"""Verify tier progression on ALL source images directly.

This script reads the source dataset and runs the CNN on all images
to get verification data without needing to generate the RL pool first.

Run:
    python training/verify_all_source.py
"""

import pandas as pd
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from gameplay.constants import DATA_DIR
from models.CNNClassifier import CNNClassifier
from models.image_transforms import green_mask, hue_proxy
from PIL import Image

def verify_all_source():
    """Load source metadata and verify all images."""
    df = pd.read_csv(_root / "data" / "consolidated_metadata.csv")
    
    print(f"Total source images: {len(df)}")
    print(f"Zombies: {len(df[df.Class == 'Zombie'])}")
    print(f"Humans (Default): {len(df[df.Class == 'Default'])}")
    print()
    
    # Initialize CNN
    clf = CNNClassifier(img_data_root=str(_root / "data"))
    
    results = []
    
    for _, row in df.iterrows():
        img_path = _root / "data" / row["Filename"]
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Get CNN prediction
            p = clf.classify_pil_image(img)
            order = p.argsort()[::-1]
            pred = int(order[0])
            p_healthy = float(p[1])
            max_prob = float(p[pred])
            margin = float(p[order[0]] - p[order[1]])
            
            # Get hue_proxy for zombies
            hue_val = 1.0
            if row["Class"] == "Zombie":
                mask = green_mask(img)
                hue_val = hue_proxy(img, mask)
            
            results.append({
                "filename": row["Filename"],
                "class": row["Class"],
                "p_healthy": p_healthy,
                "max_prob": max_prob,
                "margin": margin,
                "predicted_class": pred,
                "hue_proxy": hue_val,
            })
        except Exception as e:
            print(f"Error processing {row['Filename']}: {e}")
    
    meta = pd.DataFrame(results)
    
    # Group by class
    zombies = meta[meta["class"] == "Zombie"]
    humans = meta[meta["class"] == "Default"]
    
    print("=" * 80)
    print("ZOMBIES (all source images):")
    print("=" * 80)
    print(f"Count: {len(zombies)}")
    print(f"Mean hue_proxy: {zombies['hue_proxy'].mean():.3f}")
    print(f"Mean P(healthy): {zombies['p_healthy'].mean():.3f}")
    print(f"Mean CNN confidence: {zombies['max_prob'].mean():.3f}")
    print()
    
    print("=" * 80)
    print("HUMANS (all source images):")
    print("=" * 80)
    print(f"Count: {len(humans)}")
    print(f"Mean P(healthy): {humans['p_healthy'].mean():.3f}")
    print(f"Mean CNN confidence: {humans['max_prob'].mean():.3f}")
    print()
    
    # Save results
    meta.to_csv(_root / "data" / "source_verification.csv", index=False)
    print(f"Saved results to data/source_verification.csv")

if __name__ == "__main__":
    verify_all_source()
