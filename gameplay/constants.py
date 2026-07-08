"""
Game configuration constants.

Edit these values to design experiments and test different game configurations.
- SHIFT_LENGTH controls how much time you have per game
- AMBULANCE_CAPACITY controls how many humanoids can ride in the ambulance
- CLASSIFIER_MODEL_PATH / RL_MODEL_PATH point to your trained model weights
"""

import os

# Game rules
SHIFT_LENGTH: int = 720  # total game time in minutes
AMBULANCE_CAPACITY: int = 10  # max humanoids the ambulance can hold at once

# UI window dimensions
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 800

# Data directory — path to the folder containing consolidated_metadata.csv
DATA_DIR: str = "data"

# Model weight paths — update these to point to your trained models
CLASSIFIER_MODEL_PATH: str = os.path.join("models", "weights", "baselineCNN.pth")
RL_MODEL_PATH: str = os.path.join("models", "weights", "baselineRL.pth")

# Clock display style.
# True  = analog clock (counts up from midnight — shows elapsed time)
# False = digital countdown clock (shows remaining minutes, e.g. "342 min remaining")
#
# This is an example of the feature-flag pattern. To add your own swappable
# component: (1) define a flag here, (2) add a conditional import in the UI
# file that uses it, (3) make sure both implementations share the same public
# interface (same __init__ signature and methods).
LEGACY_CLOCK: bool = True

# Observation space size — total number of slots in the flat obs vector.
# Update this if you add or remove slots; all consumers import from here.
OBS_SIZE: int = 15

# Reward weights — must sum to 1. Adjust to shift agent priorities:
#   HEADCOUNT_WEIGHT=1.0 → pure headcount (saved − killed), ignores value
#   HEADCOUNT_WEIGHT=0.0 → pure value signal, ignores headcount
HEADCOUNT_WEIGHT: float = 0.5  # weight on (saved − killed)
VALUE_WEIGHT: float = 1.0 - HEADCOUNT_WEIGHT  # weight on saved_value

# Penalty applied when the agent attempts a blocked action (e.g. save when full).
# Negative nudge to discourage no-ops without distorting the reward scale too much.
BLOCKED_ACTION_PENALTY: float = -0.5

# CNN confidence thresholds — used to bin the max softmax probability into
# low/medium/high confidence. Shared between train_env and rl_recommender;
# change here and both inference and training stay in sync.
CNN_CONFIDENCE_MID: float = 0.5  # below this → low confidence (bin 0)
CNN_CONFIDENCE_HIGH: float = 0.75  # above this → high confidence (bin 2)

# UI color palette — adjust these to restyle the interface.
UI_COLORS: dict[str, str] = {
    "bg": "#1a1a1a",  # root and canvas background
    "panel": "#242424",  # widget panels (slightly lighter)
    "fg": "#f0f0f0",  # default text
    "btn_skip": "#2e4057",  # muted blue — neutral action
    "btn_squish": "#7b2d2d",  # dark red — destructive action
    "btn_save": "#1e4d35",  # dark green — positive action
    "btn_scram": "#6b4c11",  # amber — urgency / retreat
    "btn_default": "#333333",  # machine mode / fallback buttons
    "btn_active": "#4a4a4a",  # hover / active state
    "meter_filled": "#e63946",  # ambulance slot filled (danger red)
    "accent": "#e63946",  # highlights and end-screen text
}
