# BWSI SGAI 2026 — Cheat Sheet

**Frozen at commit `a98f719`.** Use this for quick reference during experiments.
Full reference: `docs/student-README.md`, `docs/troubleshooting.md`.

---

## CLI Commands

```bash
# Activate environment first
conda activate bwsi-sgai

# Game modes
python main.py -m human                       # human play, no weights needed
python main.py -m human_cnn                   # CNN labels humanoids; human acts
python main.py -m human_cnn_rl                # CNN + RL recommends; human acts
python main.py -m auto --log --seed 42        # headless RL run, writes summary.csv

# Training
python main.py train --type cnn               # train CNN (~2 min)
python main.py train --type rl                # train RL, 2500 ep (~20–40 min)
python main.py train --type rl -e 500         # fast RL baseline (~5–10 min)
python -u main.py train --type rl -e 500      # same, unbuffered stdout (use if silent)

# Tests
python -m pytest tests/                       # run all 293 tests
python -m pytest tests/ --cov                 # with coverage report

# Multi-seed experiment batch
for seed in 42 0 1 7 99; do python main.py -m auto --log --seed $seed; done
python -c "import pandas as pd; df = pd.read_csv('summary.csv'); print(df.tail(5)[['seed','reward','saved','killed']].to_string())"

# Reset logs before a new experiment condition
rm log.csv summary.csv
```

---

## Game Modes

| Mode | What it does | Needs CNN weights | Needs RL weights |
|------|-------------|:-----------------:|:----------------:|
| `human` | Human plays, no AI | No | No |
| `human_cnn` | CNN labels humanoids; human acts | Yes | No |
| `human_rl` | RL recommends; human acts | Yes | Yes |
| `human_cnn_rl` | CNN labels + RL recommends; human acts | Yes | Yes |
| `machine` | CNN + RL play; press button to advance turns | Yes | Yes |
| `auto` | Headless RL run; logs to `summary.csv` | Yes | Yes |

**Most complete interactive mode:** `human_cnn_rl`.
**No-weights mode for debugging:** `human` or swap in `OracleClassifier` (see Gotchas).

---

## Extension Points

| File | What to change | Effect | Retrain RL? |
|------|---------------|--------|:-----------:|
| `gameplay/constants.py` | `HEADCOUNT_WEIGHT` (0.5) | Shifts reward: headcount vs. value | Yes |
| `gameplay/constants.py` | `SHIFT_LENGTH` (720 min) | Game length. Shorter makes SCRAM proportionally more expensive | Recommended |
| `gameplay/constants.py` | `AMBULANCE_CAPACITY` (10) | Ambulance slot count | Recommended |
| `gameplay/enums.py` | `ActionCost` values | Changes action trade-offs. Keep `ActionState` in sync or changes are a silent no-op | Yes |
| `game_interfaces/data_parser.py` | `datarow_to_value()` | Triage value rule; depends on CSV columns `Injured`, `Gender`, `Item`. Return must stay in `[0.0, 1.0]` | No |
| `models/HeuristicRecommender.py` | `recommend()` rules | Rule-based strategy, no training needed. **Activate in `main.py` first** (see Gotchas) | No |
| `training/train_rl.py` | `NUM_EPISODES`, `GAMMA`, `EPSILON_DECAY` | RL training behavior | — |
| `training/train_cnn.py` | `NUM_EPOCHS` | Stay under 20 on 160-image dataset (overfitting risk) | — |

---

## Gotchas

- **`HeuristicRecommender` is not wired by default.** `RLRecommender` is the production default. To use your rule changes: edit `main.py` and replace `RLRecommender(...)` with `HeuristicRecommender()` in **both** `_make_components()` and `_run_auto()`. Then revert with `git restore main.py` when done.

- **`CNNClassifier` appears 5 times in `main.py`.** 4 in `_make_components()` (one per mode branch) + 1 in `_run_auto()`. Replacing only the `_run_auto()` occurrence means GUI modes still use CNN while `auto` mode uses your swap. Use `Ctrl+F` to find all occurrences before editing.

- **`log.csv` and `summary.csv` append across runs.** Delete both files before starting a new experiment condition. Otherwise your results mix across experiments silently. Command: `rm log.csv summary.csv`.

- **OracleClassifier diagnostic:** If the game plays well with `OracleClassifier` but poorly with `CNNClassifier`, the CNN is the bottleneck. If it plays poorly with both, the RL policy needs work. `OracleClassifier` requires no weights — swap it anywhere `CNNClassifier` is used in `main.py`.

- **RL weights only save when the agent improves.** During training, weights write only when `episode_reward > best_reward AND epsilon < 0.5`. If your run finds nothing better than the existing baseline, the weights file does not change — that is correct behavior, not a bug.

- **Comparison trap.** If one condition uses more training episodes than another, you are measuring training time, not your experimental variable. Both conditions must use the same `NUM_EPISODES`. Changing `HEADCOUNT_WEIGHT` after training also invalidates comparisons — the RL policy was trained on the old reward signal.

- **`ActionState` and `ActionCost` must stay in sync.** Adding an entry to one without the other produces a silent no-op in `map_do_action()`, not an error.

---

## Scoring Formula

```
reward = HEADCOUNT_WEIGHT × (saved − killed) + VALUE_WEIGHT × saved_value
```

**File:** `gameplay/scorekeeper.py` — look for `get_score()` and the reward calculation.
**Weights file:** `gameplay/constants.py`

| Constant | Default | Note |
|----------|---------|------|
| `HEADCOUNT_WEIGHT` | `0.5` | Weight on `(saved − killed)` |
| `VALUE_WEIGHT` | `0.5` | `1.0 - HEADCOUNT_WEIGHT` — auto-computed |
| `BLOCKED_ACTION_PENALTY` | `-0.5` | Applied when agent attempts a blocked action |

Setting `HEADCOUNT_WEIGHT = 1.0` → pure headcount signal, value ignored.
Setting `HEADCOUNT_WEIGHT = 0.0` → pure value signal, headcount ignored.

---

## Quick API

**`ScoreKeeper` — instantiation and attribute names:**

```python
from gameplay.scorekeeper import ScoreKeeper
from gameplay.constants import SHIFT_LENGTH, AMBULANCE_CAPACITY

sk = ScoreKeeper(shift_length=SHIFT_LENGTH, capacity=AMBULANCE_CAPACITY, seed=42)
sk.get_score()       # → {'killed': 0, 'saved': 0, 'saved_value': 0.0}
sk.remaining_time    # correct  (NOT time_remaining)
sk.capacity          # correct  (NOT ambulance_capacity)
sk.ambulance         # → {'zombie': 0, 'injured': 0, 'healthy': 0, 'corpse': 0}
```

> **Seed disambiguation:** `ScoreKeeper(seed=42)` records the seed in `summary.csv` only — it does NOT control humanoid ordering. `DataParser(seed=42)` controls ordering and reproducibility. Use `DataParser` for reproducible experiments.

**`probs_to_class` — convert classifier output to a label:**

```python
from models.HumanoidClassifier import probs_to_class

probs = classifier.classify(humanoid)
# probs order: [zombie, healthy, injured, corpse] — do NOT reorder State entries
label, confidence = probs_to_class(probs)
# label: State.HEALTHY | State.ZOMBIE | State.INJURED | State.CORPSE
# confidence: float in [0.0, 1.0]
# Do NOT use np.argmax(probs) directly — returns int, not State
```

**`datarow_to_value` touch cell — requires DataFrame setup:**

```python
import pandas as pd
from game_interfaces.data_parser import datarow_to_value

df = pd.read_csv("data/consolidated_metadata.csv")
row = df.iloc[0]
value = datarow_to_value(row)   # default: random float in [0.0, 1.0]
print(value)
# Use rng.random() (not random.random()) inside datarow_to_value for reproducibility
```

**OracleClassifier swap — diagnostic pattern:**

```python
# In main.py, replace:
from models.CNNClassifier import CNNClassifier
# with:
from models.OracleClassifier import OracleClassifier

# Then replace CNNClassifier(...) with OracleClassifier() — no arguments, no weights file needed
# 5 occurrences in main.py total (use Ctrl+F)
# Revert: git restore main.py
```

---

## Where to Look

| I want to... | Open this file |
|---|---|
| Change the reward formula weights | `gameplay/constants.py` → `HEADCOUNT_WEIGHT` |
| Change how long the game runs | `gameplay/constants.py` → `SHIFT_LENGTH` |
| Change action time costs | `gameplay/enums.py` → `ActionCost` |
| Change how triage value is computed | `game_interfaces/data_parser.py` → `datarow_to_value()` |
| Write a rule-based decision strategy | `models/HeuristicRecommender.py` → `recommend()` |
| Change RL training hyperparameters | `training/train_rl.py` → `NUM_EPISODES`, `GAMMA`, `EPSILON_DECAY` |
| Change CNN training hyperparameters | `training/train_cnn.py` → `NUM_EPOCHS` |
| Read experiment results | `summary.csv` (one row per run), `log.csv` (one row per turn) |
| Understand the observation vector | `gameplay/constants.py` → `OBS_SIZE`; `game_interfaces/train_env.py` |
| Find all tunable constants | `gameplay/constants.py` — start here for every experiment |
| Troubleshoot a crash or env issue | `docs/troubleshooting.md` |
