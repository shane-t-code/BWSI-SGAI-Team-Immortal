# Student Guide — BWSI SGAI 2026

You've run the game at least once. This guide tells you where everything lives and what you can actually change. By the end, you should be able to open the right file in under 30 seconds for any common experiment.

---

## How the game works (the 30-second version)

You're driving an ambulance through a zombie apocalypse. Each turn, you see a humanoid and pick an action:

- **Save** — load them onto the ambulance (costs 30 min)
- **Skip** — leave them behind (costs 15 min)
- **Squish** — destroy them on the spot (costs 5 min)
- **Scram** — drive to the hospital and unload everyone (costs 120 min)

The catch: if any zombie is on the ambulance when you scram, *everyone on board dies*. You have 720 minutes of shift time. When time runs out, the game ends and you get a score.

Your score is:

```
score = 0.5 × (saved − killed) + 0.5 × saved_value
```

Both weights are `0.5` by default. Changing them is one of the most common experiments you'll run.

---

## Where stuff lives

| Path | What's in it |
|---|---|
| `main.py` | Entry point — picks game mode and wires everything together |
| `gameplay/` | Core game logic — rules, scoring, constants |
| `game_interfaces/` | Connects the game to the data and the RL training loop |
| `models/` | CNN classifier, RL recommender, DQN agent |
| `ui/` | Everything you see on screen (Tkinter) |
| `training/` | Scripts and notebooks for training models |
| `data/` | Images and the metadata CSV |
| `tests/` | Automated tests (pytest) |
| `docs/` | Guides and architecture notes — you are here |
| `lectures/` | Jupyter notebooks for each module |

---

## Running the game

Pick one mode and run it — these are alternatives, not a sequence. Start with `human` to learn the game, then try the others.

| Mode | What you experience |
|---|---|
| `human` | You classify every humanoid and choose every action |
| `human_cnn` | The CNN shows its prediction — you make the final call |
| `human_rl` | The RL agent suggests an action — you decide whether to follow it |
| `human_cnn_rl` | Both models advise you; you decide |
| `machine` | CNN + RL play together; you press a button to advance each turn |
| `auto` | RL plays silently with no window — score prints when done |

```
python main.py -m human
```

Replace `human` with any mode name from the table above.

Two useful flags you can add to any mode:

- `--log` — saves every turn to `log.csv` and `summary.csv`
- `--seed 42` — fixes the humanoid order so runs are reproducible

```
python main.py -m auto --log --seed 42
```

### Training

The notebooks in `training/` are the easiest way to train — they show loss curves and let you experiment interactively. Or from the terminal:

```
python main.py train --type cnn
python main.py train --type rl
```

---

## The five things you're most likely to change

### 1. Change the scoring formula

**File:** `gameplay/constants.py`

```python
HEADCOUNT_WEIGHT: float = 0.5   # weight on (saved − killed)
VALUE_WEIGHT: float = 1.0 - HEADCOUNT_WEIGHT  # weight on saved_value
```

Change `HEADCOUNT_WEIGHT` and `VALUE_WEIGHT` updates automatically. Try extremes:

- `HEADCOUNT_WEIGHT = 1.0` — pure headcount, ignores value entirely
- `HEADCOUNT_WEIGHT = 0.0` — pure value, ignores headcount entirely

After changing this, retrain the RL agent (`python main.py train --type rl`) so it learns under the new scoring rules. The old weights were trained on the old formula.

---

### 2. Change action costs

**File:** `gameplay/constants.py` → `gameplay/enums.py`

Action costs live in `gameplay/enums.py`:

```python
class ActionCost(Enum):
    SAVE   = 30    # minutes
    SQUISH = 5
    SKIP   = 15
    SCRAM  = 120
```

Making skip cheaper than squish changes the trade-off the agent faces. After changing costs, retrain the RL agent so it learns the new timing.

Note: `ActionCost` and `ActionState` must stay in sync — if you add a new action to one, add it to both.

---

### 3. Change how humanoids are assigned a value

**File:** `game_interfaces/data_parser.py` → `datarow_to_value()`

By default, every humanoid gets a random value between 0 and 1. You can make value depend on actual data:

```python
def datarow_to_value(data_row: pd.Series, rng=None) -> float:
    # data_row columns available: Class, Injured, Gender, Item, Filename
    if data_row["Injured"]:
        return 0.9   # injured humans are always high priority
    return 0.3
```

The `data_row` is one row from `data/consolidated_metadata.csv`. Open that file in a spreadsheet to see what columns are available.

---

### 4. Change what the CNN classifies

**File:** `models/CNNClassifier.py` + `training/train_cnn.ipynb`

The CNN reads a humanoid image and outputs probabilities for four classes: zombie, healthy, injured, corpse. To change how it classifies, edit `CNNClassifier.py` and retrain with the notebook.

The trained weights are saved to `models/weights/baselineCNN.pth`. The path is set in `gameplay/constants.py`:

```python
CLASSIFIER_MODEL_PATH: str = os.path.join("models", "weights", "baselineCNN.pth")
```

---

### 5. Change how the RL agent decides

**File:** `models/RLRecommender.py` + `training/train_rl.ipynb`

The RL agent takes a 15-number observation vector and picks an action. To change its behavior, you can:

- **Tune hyperparameters** — edit the constants at the top of `training/train_rl.py`:
  ```python
  NUM_EPISODES = 2500   # more = longer training, usually better
  GAMMA = 0.99          # how much the agent values future rewards (0–1)
  LEARNING_RATE = 1e-3  # step size for gradient updates
  ```
- **Change the reward shaping** — edit `game_interfaces/train_env.py` (the `step()` method computes per-turn reward as the delta in cumulative score)
- **Change the scoring formula** — see #1 above; the reward the agent sees uses `HEADCOUNT_WEIGHT` and `VALUE_WEIGHT` directly

After any change, retrain: `python main.py train --type rl`

---

## What ScoreKeeper tracks

`ScoreKeeper` is the game's bookkeeper. It lives in `gameplay/scorekeeper.py`. You don't need to edit it to run experiments, but knowing what it tracks helps you understand the data.

| Attribute | What it is |
|---|---|
| `sk.remaining_time` | Minutes left in the shift |
| `sk.capacity` | Max humanoids the ambulance can hold (10 by default) |
| `sk.ambulance` | Dict: how many of each type are currently on board |

To read the score at any point:

```python
result = sk.get_score()
# result = {"killed": 3, "saved": 7, "saved_value": 4.2}
```

`get_score()` scrams anyone still on the ambulance before returning the final tally — so the number you get is always complete.

---

## Logging and analysis

Add `--log` to any run to write two CSV files:

```bash
python main.py -m auto --log
```

**`log.csv`** — one row per turn. Columns include:
- `action` — what action was taken
- `humanoid_ground_truth` — the actual state (zombie/healthy/injured/corpse)
- `humanoid_predicted_class` — what the CNN predicted (as an integer: 0=zombie, 1=healthy, 2=injured, 3=corpse)
- `confidence` — CNN confidence bucket: `"low"`, `"mid"`, or `"high"`
- `remaining_time` — time left before this action
- `humanoid_value` — the humanoid's value score

**`summary.csv`** — one row per game. Columns include:
- `killed`, `saved`, `saved_value` — final score
- `reward` — the weighted reward the RL agent optimized for
- `num_turns` — how many actions were taken

Both files use a `run_id` column (a UUID) to link turns to their game. To join them in pandas:

```python
import pandas as pd

log = pd.read_csv("log.csv")
summary = pd.read_csv("summary.csv")
combined = log.merge(summary, on="run_id")
```

Both files **append** across runs — each new game adds rows. Delete the files to start fresh.

---

## What the humanoid states mean

| State | What it means | Save outcome | Skip outcome | Squish outcome |
|---|---|---|---|---|
| `healthy` | Uninfected human | Boards ambulance | Left behind (no kill) | Counted as killed |
| `injured` | Hurt human | Boards ambulance | Counted as killed | Counted as killed |
| `zombie` | Infected | Boards ambulance (danger!) | Left behind (no kill) | No score effect |
| `corpse` | Dead zombie | Takes a slot, discarded at scram | Left behind (no kill) | No score effect |

The tricky cases:
- Saving a zombie and then scramming kills everyone else on the ambulance
- Skipping an injured human counts as a kill (they won't survive without rescue)
- Squishing a zombie costs 5 min but doesn't affect your score

States are defined in `gameplay/enums.py`. The mapping from image data to state is in `game_interfaces/data_parser.py` → `datarow_to_state()`.

---

## The observation vector (if you're working on RL)

The RL agent doesn't see the game the way you do. It sees a flat list of 15 numbers, all between 0 and 1. Built in `game_interfaces/obs_builder.py`; size set by `OBS_SIZE = 15` in `gameplay/constants.py`.

| Slot(s) | What it encodes | Example |
|---|---|---|
| 0 | Time remaining — one of five steps: 0.0, 0.25, 0.5, 0.75, 1.0 (1.0 = full shift, 0.0 = almost out) | `0.50` (mid-shift) |
| 1–4 | Count of each type on the ambulance right now (zombie, healthy, injured, corpse) | `[0, 3, 1, 0]` (3 healthy, 1 injured on board) |
| 5–8 | CNN probabilities for this humanoid (zombie, healthy, injured, corpse) | `[0.85, 0.05, 0.08, 0.02]` (CNN thinks zombie) |
| 9 | CNN confidence — one of three steps: 0.0 = low, 0.5 = mid, 1.0 = high | `1.0` (high confidence) |
| 10–13 | Which actions are available right now (1=yes, 0=no): save, skip, squish, scram | `[0, 1, 1, 1]` (ambulance full — save blocked) |
| 14 | Humanoid value — one of five steps: 0.0, 0.25, 0.5, 0.75, 1.0 (1.0 = highest priority) | `0.75` (value between 0.6 and 0.8) |

**How values get binned:** continuous values aren't passed in raw — they're quantized into 5 levels (0–4) and then divided by 4 to land on {0.0, 0.25, 0.5, 0.75, 1.0}. CNN confidence uses 3 levels instead: top prob ≥ 0.75 → high (1.0), ≥ 0.5 → mid (0.5), else low (0.0). This coarsening is intentional — it makes the state space small enough for the DQN to learn from.

If you add a new slot, increment `OBS_SIZE` in `constants.py` and update the RL model accordingly — the trained weights won't work with a different vector length.

---

**Example 1 — Easy save, early shift**

10 minutes in. Ambulance has 2 healthy on board, 8 slots free. New arrival looks healthy: CNN top prob 0.92, value 0.6.

| Slot(s) | What | Value |
|---|---|---|
| 0 | Time remaining | 1.00 — bin 4 (full shift) |
| 1–4 | Ambulance: zombie, healthy, injured, corpse | 0, 2, 0, 0 |
| 5–8 | CNN probs: zombie, healthy, injured, corpse | 0.03, 0.92, 0.04, 0.01 |
| 9 | CNN confidence | 1.00 — high (0.92 ≥ 0.75) |
| 10–13 | Actions: save, skip, squish, scram | 1, 1, 1, 1 — all available |
| 14 | Humanoid value | 0.75 — bin 3 (value 0.6) |

`obs = [1.00, 0, 2, 0, 0, 0.03, 0.92, 0.04, 0.01, 1.00, 1, 1, 1, 1, 0.75]`

No threat, plenty of time, CNN is confident — a trained agent saves.

---

**Example 2 — Tense: probable zombie, ambulance nearly full**

Mid-shift, ~360 min left. Ambulance holds 9 humanoids (7 healthy, 1 injured, 1 corpse) — one slot remains. New arrival: CNN says zombie at 0.88, value 0.1.

| Slot(s) | What | Value |
|---|---|---|
| 0 | Time remaining | 0.50 — bin 2 (half shift) |
| 1–4 | Ambulance: zombie, healthy, injured, corpse | 0, 7, 1, 1 |
| 5–8 | CNN probs: zombie, healthy, injured, corpse | 0.88, 0.06, 0.04, 0.02 |
| 9 | CNN confidence | 1.00 — high (0.88 ≥ 0.75) |
| 10–13 | Actions: save, skip, squish, scram | 1, 1, 1, 1 — one slot still free |
| 14 | Humanoid value | 0.00 — bin 0 (value 0.1) |

`obs = [0.50, 0, 7, 1, 1, 0.88, 0.06, 0.04, 0.02, 1.00, 1, 1, 1, 1, 0.00]`

Saving a probable zombie risks killing 9 people at scram. Low value, high-confidence threat — a trained agent squishes or skips.

---

**Example 3 — Uncertain: low CNN confidence, high-value arrival**

Late shift, ~180 min left. Ambulance has 4 healthy on board. CNN is unsure — top prob only 0.45. Value is high at 0.8.

| Slot(s) | What | Value |
|---|---|---|
| 0 | Time remaining | 0.25 — bin 1 (late shift) |
| 1–4 | Ambulance: zombie, healthy, injured, corpse | 0, 4, 0, 0 |
| 5–8 | CNN probs: zombie, healthy, injured, corpse | 0.30, 0.20, 0.45, 0.05 |
| 9 | CNN confidence | 0.00 — low (top prob 0.45 < 0.50) |
| 10–13 | Actions: save, skip, squish, scram | 1, 1, 1, 1 — all available |
| 14 | Humanoid value | 1.00 — bin 4 (value 0.8) |

`obs = [0.25, 0, 4, 0, 0, 0.30, 0.20, 0.45, 0.05, 0.00, 1, 1, 1, 1, 1.00]`

High value but the CNN has no idea what it's looking at. Saving risks 4 lives if it's a zombie — this is where the agent's learned risk tolerance matters most.

---

## Quick reference

| Goal | File |
|---|---|
| Change scoring weights | `gameplay/constants.py` |
| Change action time costs | `gameplay/enums.py` |
| Change game length or ambulance size | `gameplay/constants.py` |
| Change how humanoid value is assigned | `game_interfaces/data_parser.py` |
| Change how CSV columns map to states | `game_interfaces/data_parser.py` |
| Tune RL hyperparameters | `training/train_rl.py` |
| Retrain the RL agent | `training/train_rl.ipynb` |
| Retrain the CNN | `training/train_cnn.ipynb` |
| Understand the RL observation space | `game_interfaces/obs_builder.py` |
| Add a new humanoid state | `gameplay/enums.py` + `gameplay/scorekeeper.py` |
| Change UI colors | `gameplay/constants.py` (`UI_COLORS`) |
| Read the full design doc | `docs/game-design.md` |
