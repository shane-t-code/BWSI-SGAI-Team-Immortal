# Dead or Alive 2026 — Game Design Document
**Version:** 1.0
**Date:** 2026-04-05
**Status:** Authoritative — all code changes, tests, and lecture content should conform to this document.

---

## 1. Purpose and Pedagogical Goals

*Dead or Alive 2026* is a zombie-apocalypse triage game used in the MIT Beaverworks Summer Institute (BWSI) Serious Games with AI (SGAI) course. It gives you a working baseline AI system to study, extend, and redesign. The game is not meant to be a polished product — it is a concrete, playable system you can interrogate, break, and improve.

**Core pedagogical goals:**

- **Classification under uncertainty.** Humanoids must be identified before acting on them, but the classifier is imperfect. You'll experience first-hand why uncertainty matters and what it costs when a model is wrong.
- **Human-machine teaming.** The game supports modes where a human and machine share decision-making. You'll develop intuition for when to trust the machine and when to override it.
- **Reward shaping.** The default reward (`saved − killed`) is a starting point, not an answer. You'll encounter situations where a single number fails to capture what you care about, motivating you to design better reward functions.
- **Full ML pipeline.** You'll train a CNN classifier and a DQN agent, integrate both into a running game, and evaluate their behavior — experiencing the complete arc from data to deployment.
- **RL environment design.** The observation space, action space, and reward function are all visible and modifiable. You'll learn how these design choices shape what an agent learns.
- **Confidence and uncertainty.** The CNN produces a confidence score alongside its classification. You'll work with this signal — deciding whether a low-confidence prediction is worth trusting.
- **Iterative development on a baseline.** You'll extend a working system rather than building from scratch. This mirrors real-world ML engineering practice.
- **Competing objectives.** The game reports multiple metrics with no combined score. You — and your machines — must decide what "good" looks like, and often disagree. That disagreement is part of the lesson.
- **Responsibility in automated systems.** When the machine makes a wrong call, who bears responsibility? The game puts you in that position repeatedly.

---

## 2. Game Overview

The player works as an emergency responder during a zombie outbreak. Humanoids appear one at a time. The player must classify each humanoid and decide what to do: save them (load them into the ambulance), squish them, skip them, or scram (drive the ambulance back to the hospital to drop off passengers).

The game ends when the player runs out of time or all humanoids in the dataset have been encountered. At the end, the game reports how many humans were saved and how many were killed.

There is no win condition. The game ends and reports metrics. What counts as a good outcome is left for the player to decide.

---

## 3. Humanoid Types

Humanoids are drawn from a dataset with two metadata fields: `Class` (Default or non-Default) and `Injured` (boolean). These combine to produce four states:

| State | Class | Injured | Description |
|-------|-------|---------|-------------|
| HEALTHY | Default | False | A healthy, uninjured human. |
| INJURED | Default | True | A human who has been injured. Cannot survive without being saved. |
| ZOMBIE | Non-Default | False | An active zombie. Dangerous if loaded into the ambulance. |
| CORPSE | Non-Default | True | A dead zombie. No longer dangerous. Can be saved — takes up an ambulance slot (wasting capacity) and is discarded at scram with zero score credit. |

**Key rules:**
- HEALTHY and INJURED apply only to humans. ZOMBIE and CORPSE apply only to zombies. There are no human corpses.
- A CORPSE is a zombie that is already dead — it cannot spread infection or attack.
- The CNN classifier produces a prediction and confidence score for each humanoid. The classifier may be wrong. This uncertainty is the core tension of the game.

### 3a. Humanoid Value

Each humanoid has a `value` attribute (float in `[0.0, 1.0)`) assigned at creation by `DataParser`. Value represents triage priority: a higher value means higher rescue priority.

Value is displayed to the player during gameplay as both the raw float and a category label:

| Range | Label |
|-------|-------|
| `[0.0, 0.2)` | VERY LOW |
| `[0.2, 0.4)` | LOW |
| `[0.4, 0.6)` | NEUTRAL |
| `[0.6, 0.8)` | HIGH |
| `[0.8, 1.0)` | VERY HIGH |

Value assignment is yours to extend. The default logic lives in `datarow_to_value()` in `game_interfaces/data_parser.py` — replace or modify that function to implement priority schemes based on injury status, class, or any other signal in the dataset row.

---

## 4. Actions

Four actions are available each turn. Each action costs time (in minutes).

| Action | Time Cost | Effect |
|--------|-----------|--------|
| SAVE | 30 min | Loads the humanoid into the ambulance. |
| SKIP | 15 min | Does nothing. The humanoid is passed over. |
| SQUISH | 5 min | Eliminates the humanoid. |
| SCRAM | 120 min | Drives the ambulance back to the hospital, resolving all passengers. |

### Action consequences by humanoid type

| Action | HEALTHY | INJURED | ZOMBIE | CORPSE |
|--------|---------|---------|--------|--------|
| SAVE | Added to ambulance (healthy slot) | Added to ambulance (injured slot) | Added to ambulance (zombie slot) — **dangerous** | Added to ambulance (corpse slot) — takes a slot, harmless |
| SKIP | No consequence | **Killed** (left to die) | No consequence | No consequence |
| SQUISH | **Killed** | **Killed** | Neutral (no kill credit, no reward) | Neutral (no kill credit, no reward) |
| SCRAM | Ambulance resolves — see Section 6 | Ambulance resolves — see Section 6 | Ambulance resolves — poisons if any zombie present | Discarded (neutral) |

**Notes:**
- Skipping an injured human kills them — they cannot survive without help.
- Squishing a zombie is neutral: it clears a threat but earns no score. The squished counter is tracked; see Section 7.
- Saving a zombie loads it into the ambulance. If the ambulance is scrammed with a zombie aboard, all non-zombie passengers are killed (see Section 6). Saving a zombie is always a strategic mistake, but it is permitted.
- Saving a corpse wastes ambulance capacity but has no other negative effect. Corpses do not trigger the zombie kill rule.

---

## 5. The Ambulance

The ambulance is a shared resource with a fixed capacity (default: 10).

**Ambulance state:** The ambulance tracks four passenger counts:
- `zombie` — number of zombies loaded
- `injured` — number of injured humans loaded
- `healthy` — number of healthy humans loaded
- `corpse` — number of corpses loaded

**Capacity:** `at_capacity()` is true when the sum of all four counts ≥ `AMBULANCE_CAPACITY`. A corpse takes up a real slot. When at capacity, SAVE is disabled.

**After scram:** All four counts are reset to 0. A new humanoid appears and the game continues (as long as time > 0 and humanoids remain).

---

## 6. Scram Resolution

When the player scramas (manually or via auto-scram at game end):

1. **If `ambulance["zombie"] > 0`:**
   - All injured and healthy passengers are killed: `killed += ambulance["injured"] + ambulance["healthy"]`

2. **If `ambulance["zombie"] == 0`:**
   - All injured and healthy passengers are saved: `saved += ambulance["injured"] + ambulance["healthy"]`

3. **Corpses are always discarded** (neutral — already dead, no save or kill credit). Corpses in the ambulance are irrelevant to the zombie kill-rule check — only `ambulance["zombie"] > 0` triggers the kill rule.

4. Ambulance is cleared. If `remaining_time > 0` and humanoids remain, a new humanoid appears.

**Empty ambulance scram:** Permitted. Costs 120 minutes with no passengers to resolve. This may happen accidentally or from a poor recommendation.

**Auto-scram at game end:** When the game ends, if the ambulance is not empty, `get_score()` triggers an automatic scram. This is thematically correct — the ambulance always returns to the hospital. There is no penalty for not scraming manually before time runs out.

---

## 7. Scoring

Three metrics are tracked in `ScoreKeeper`:

| Metric | What it counts | Shown at game end? |
|--------|---------------|-------------------|
| `saved` | Total injured + healthy passengers delivered safely (no zombie aboard at scram) | Yes |
| `killed` | Total humans killed: squish a healthy or injured human; skip an injured human; scram with a zombie aboard (kills all injured + healthy passengers) | Yes |
| `squished` | Total squish actions taken regardless of humanoid type (note: squishing a HEALTHY or INJURED human also increments `killed`) | No — tracked in code only; see Section 15 |

**Reward for RL training:** `get_cumulative_reward()` returns a **linear combination** of two signals:

```
reward = HEADCOUNT_WEIGHT × (saved − killed) + VALUE_WEIGHT × saved_value
```

A linear combination means each term is multiplied by a weight and the results are added together — the total reward is the weighted sum of two signals. `HEADCOUNT_WEIGHT` and `VALUE_WEIGHT` control how much each signal contributes. **They must sum to `1.0`** to keep the reward scale consistent: weights that don't add to 1.0 inflate or deflate the magnitude of every reward signal, which destabilises RL training. Default is both equal to `0.5` (equal weight) — if `HEADCOUNT_WEIGHT = 0.8` and `VALUE_WEIGHT = 0.2`, the headcount signal dominates; if `HEADCOUNT_WEIGHT = 0.2` and `VALUE_WEIGHT = 0.8`, the agent prioritises accumulated value over raw body count.

Tune these in `gameplay/constants.py` — look for `HEADCOUNT_WEIGHT` and `VALUE_WEIGHT`. `VALUE_WEIGHT` is defined as `1.0 - HEADCOUNT_WEIGHT` so the two always sum to `1.0` automatically.

The step reward the agent receives is the *delta* between two calls: `reward = get_cumulative_reward() - previous`. This means a single action can produce a large swing — saving a zombie that contaminates 3 passengers shifts `saved_count − killed_count` by −6 in one step.

`saved_value` is the running sum of `humanoid.value` for all successfully delivered humanoids — i.e., non-zombie-contaminated passengers at each scram. If the ambulance contains a zombie at scram time, the value of all injured and healthy passengers aboard is lost along with their lives.

`get_score()` returns three keys:

| Key | Description |
|-----|-------------|
| `killed` | Total humans killed |
| `saved` | Total humans saved (headcount) |
| `saved_value` | Cumulative value of all saved humanoids |

**No win condition.** The game reports `saved`, `killed`, and `saved_value` as separate metrics. There is no combined score, ranking, or win/lose state. You must decide for yourself which metrics matter more.

---

## 8. Button Availability

Buttons are enabled or disabled based on current game state:

| Button | Disabled when |
|--------|--------------|
| SAVE | `remaining_time ≤ 0` OR `at_capacity()` |
| SKIP | `remaining_time ≤ 0` |
| SQUISH | `remaining_time ≤ 0` |
| SCRAM | Never disabled **during active play** — always available while the game is running, including with an empty ambulance. Disabled at game-over along with all other buttons (§9). |

There is no time-reservation constraint. Any available action may be taken at any time. Auto-scram at game end resolves the ambulance regardless of whether the player manually scrammed.

---

## 9. Game Flow

### Start
The game begins immediately when the window opens. The first humanoid is shown with no ready-up screen or countdown. In modes with a classifier, classification runs for the first humanoid on launch.

### Turn loop
1. A humanoid is shown (image + optional CNN classification + optional RL recommendation)
2. Player takes an action (or presses Advance Turn in machine mode)
3. `remaining_time` decreases by the action cost
4. If the action was SCRAM: scram resolution runs (Section 6); if the game continues, next humanoid appears
5. End conditions are checked (Section 9b)
6. If the game continues: next humanoid appears; classifier and recommender refresh for the new humanoid

### End conditions
Two end conditions exist. **Time's Up takes priority** if both are triggered on the same action.

| Condition | Trigger | End message |
|-----------|---------|-------------|
| Time's Up | `remaining_time ≤ 0` after any action | **"Time's Up"** |
| The City Is Empty | `len(data_parser.unvisited) == 0` | **"The City Is Empty"** |

Both conditions:
1. Trigger auto-scram if the ambulance is not empty
2. Display the final scores (`saved`, `killed`)
3. Disable all buttons

"The City Is Empty" is a valid end condition that becomes more likely if `SHIFT_LENGTH` is increased or a small dataset is used.

### Post-scram continuation
After a manual scram that does not end the game, a new humanoid appears immediately. Multiple scramas per game are normal and expected. The player keeps going until time runs out or the city is empty.

---

## 10. Clock

Two clock implementations are available, controlled by `LEGACY_CLOCK` in `gameplay/constants.py`.

### Analog clock (`LEGACY_CLOCK = True`, default)
Shows the current time in a 12-hour shift. The shift begins at midnight (0:00) and the clock advances toward noon (12:00) as time is spent. The clock shows elapsed time, not remaining time.

### Digital countdown clock (`LEGACY_CLOCK = False`)
Shows remaining time in minutes, counting down toward zero (e.g., "342 min remaining"). Implemented in `ui/digital_clock.py` with the same public interface as `ui/clock.py`.

The `LEGACY_CLOCK` constant is an example of the feature-flag pattern. The comment in `constants.py` explains how to replicate this pattern for other swappable components.

---

## 11. Game Modes

Six playable game modes. Training is a separate utility, not a game mode — see Section 12.

| Mode | Machine Assistance | Description | Best Used For |
|------|--------------------|-------------|---------------|
| `human` | None | Pure human play — the player sees the humanoid image and chooses an action. | Learning the rules; playing without any ML |
| `human_cnn` | CNN classifies | The CNN runs on each humanoid and displays its classification and confidence. The player decides the action. | Testing a trained CNN in the game context; observing misclassifications |
| `human_rl` | RL recommends | The RL agent suggests an action each turn. The player decides whether to follow it. No CNN classification shown. | Human-machine teaming with RL only; studying agent recommendations without CNN noise |
| `human_cnn_rl` | CNN classifies + RL recommends | The CNN identifies the humanoid and the RL agent suggests an action. The player decides whether to follow the suggestion. | Human-machine teaming; evaluating both models together; studying when to trust the machine |
| `machine` | CNN classifies + RL acts | The player presses "Advance Turn." The CNN classifies the humanoid; the RL agent selects and executes an action automatically, one step at a time. | Watching a trained agent play; debugging model behavior; demonstration |
| `auto` | CNN classifies + RL acts (headless) | Fully headless — no GUI. The CNN classifies each humanoid and the RL agent selects every action. Runs until the dataset is exhausted or time expires, whichever comes first. Prints `RL equiv reward: {float}` followed by a score dict. | Fast benchmarking; generating `log.csv`; evaluation without GUI overhead |

---

## 12. Training Utilities

Training is a separate utility, not a game mode. It is invoked via the `train` subparser rather than the `-m` flag.

```
python main.py train --type rl                                  # train DQN agent (defaults)
python main.py train --type cnn                                 # train CNN classifier (defaults)
python main.py train --type rl -e 2500 --seed 42 -o models/weights/custom.pt
```

Two underlying training scripts are also available directly:

| Script | Purpose |
|--------|---------|
| `training/train_cnn.py` (and `training/train_cnn.ipynb`) | Trains the CNN image classifier on the humanoid dataset |
| `training/train_rl.py` | Trains the DQN RL agent on the game environment |

Trained weights are saved to `models/weights/`:
- `baselineCNN.pth` — CNN classifier weights
- `baselineRL.pth` — DQN agent weights

---

## 13. RL Observation Space

The RL agent receives a 15-dimensional observation vector at each step, encoded as `spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)`. All values are normalized to `[0, 1]`. The choice of `Box` over `MultiDiscrete` is deliberate — see the inline comment in `game_interfaces/base_env.py` for the rationale.

| Index | Name | Encoding |
|-------|------|----------|
| `[0]` | `time_bin` | Shift divided into 5 equal windows (0=critical, 4=fresh), normalized ÷ 4 → `{0.0, 0.25, 0.5, 0.75, 1.0}` |
| `[1]` | `class_counts[ZOMBIE]` | `ambulance["zombie"] / AMBULANCE_CAPACITY` |
| `[2]` | `class_counts[HEALTHY]` | `ambulance["healthy"] / AMBULANCE_CAPACITY` |
| `[3]` | `class_counts[INJURED]` | `ambulance["injured"] / AMBULANCE_CAPACITY` |
| `[4]` | `class_counts[CORPSE]` | `ambulance["corpse"] / AMBULANCE_CAPACITY` |
| `[5–8]` | `humanoid_class` | One-hot over 4 classes (CNN argmax): ZOMBIE, HEALTHY, INJURED, CORPSE |
| `[9]` | `humanoid_confidence` | CNN top-prediction confidence bin: low=0.0, mid=0.5, high=1.0 |
| `[10]` | `doable_actions[SAVE]` | `1.0` if save is currently available, else `0.0` |
| `[11]` | `doable_actions[SQUISH]` | `1.0` if squish is currently available, else `0.0` |
| `[12]` | `doable_actions[SKIP]` | `1.0` if skip is currently available, else `0.0` |
| `[13]` | `doable_actions[SCRAM]` | Always `1.0` — scram is never disabled |
| `[14]` | `humanoid_value_bin` | 5-bin discretization of `humanoid.value`, normalized ÷ 4 → `{0.0, 0.25, 0.5, 0.75, 1.0}` |

### Example Observation Walkthrough

**Hypothetical game state:**
- 515 minutes left in a 720-minute shift (~71.5% remaining — plenty of time)
- Ambulance: 0 zombies, 3 healthy, 2 injured, 0 corpses (capacity = 10)
- Current humanoid: CNN predicts INJURED with confidence 0.82
- Current humanoid value: 0.73
- All actions available (time > 0, ambulance not full)

**Slot-by-slot encoding:**

| Index | Slot | Formula | This example | Result |
|-------|------|---------|--------------|--------|
| `[0]` | `time_bin` | `min(int(remaining_time / shift_length × 5), 4) ÷ 4` — `int()` truncates first, then `min()` caps at 4 | `min(int(515/720 × 5), 4) = min(int(3.576), 4) = min(3, 4) = 3` → `3 ÷ 4` | **0.75** |
| `[1]` | `class_counts[ZOMBIE]` | `ambulance["zombie"] / capacity` | `0 / 10` | **0.0** |
| `[2]` | `class_counts[HEALTHY]` | `ambulance["healthy"] / capacity` | `3 / 10` | **0.3** |
| `[3]` | `class_counts[INJURED]` | `ambulance["injured"] / capacity` | `2 / 10` | **0.2** |
| `[4]` | `class_counts[CORPSE]` | `ambulance["corpse"] / capacity` | `0 / 10` | **0.0** |
| `[5]` | `humanoid_class[ZOMBIE]` | `1 if argmax(cnn_probs) == 0 else 0` | Predicted INJURED (index 2) ≠ 0 | **0.0** |
| `[6]` | `humanoid_class[HEALTHY]` | `1 if argmax(cnn_probs) == 1 else 0` | Predicted INJURED (index 2) ≠ 1 | **0.0** |
| `[7]` | `humanoid_class[INJURED]` | `1 if argmax(cnn_probs) == 2 else 0` | Predicted INJURED (index 2) == 2 → 1 | **1.0** |
| `[8]` | `humanoid_class[CORPSE]` | `1 if argmax(cnn_probs) == 3 else 0` | Predicted INJURED (index 2) ≠ 3 | **0.0** |
| `[9]` | `humanoid_confidence` | `bin(max(cnn_probs)) ÷ 2` where bin is a custom label (not Python's built-in `bin()`): `p < 0.5 → 0`, `0.5 ≤ p < 0.75 → 1`, `p ≥ 0.75 → 2` | `0.82 ≥ 0.75` → bin 2 (high) → `2 ÷ 2` | **1.0** |
| `[10]` | `doable_actions[SAVE]` | `1 if time > 0 and not at_capacity else 0` | Time > 0, 5 of 10 slots used → available | **1.0** |
| `[11]` | `doable_actions[SQUISH]` | `1 if time > 0 else 0` | Time > 0 → available | **1.0** |
| `[12]` | `doable_actions[SKIP]` | `1 if time > 0 else 0` | Time > 0 → available | **1.0** |
| `[13]` | `doable_actions[SCRAM]` | Always `1` | Always available | **1.0** |
| `[14]` | `humanoid_value_bin` | `min(int(value × 5), 4) ÷ 4` — same pattern as `time_bin`: `int()` truncates first, then `min()` caps at 4 | `min(int(0.73 × 5), 4) = min(3, 4) = 3` → `3 ÷ 4` | **0.75** |

**Resulting obs vector:**
```
[0.75, 0.0, 0.3, 0.2, 0.0, | 0.0, 0.0, 1.0, 0.0, | 1.0, | 1.0, 1.0, 1.0, 1.0, | 0.75]
  time   ←class_counts→      ←humanoid_class (1-hot)→  conf   ←doable_actions→    value
```

Note that `humanoid_class` uses **one-hot encoding** rather than a raw integer (e.g. `2` for INJURED). A raw integer implies ordinal meaning — that INJURED (2) is "twice" HEALTHY (1) — which is false. One-hot gives each class its own independent binary feature, removing that false relationship. The same applies to the State enum ordering in slots [1–4].

> **Migration note:** Existing RL weights trained on the previous 14-dimensional observation vector are **incompatible** with this 15-dimensional space. You'll need to retrain the DQN agent from scratch. The old `baselineRL.pth` will produce a shape mismatch error on load.

---

## 14. Configuration Constants

All tunable game parameters live in `gameplay/constants.py`.

| Constant | Default | Description |
|----------|---------|-------------|
| `SHIFT_LENGTH` | 720 | Total game time in minutes |
| `AMBULANCE_CAPACITY` | 10 | Maximum humanoids the ambulance can hold at once |
| `WINDOW_WIDTH` | 1280 | UI window width in pixels |
| `WINDOW_HEIGHT` | 800 | UI window height in pixels |
| `DATA_DIR` | `"data"` | Path to the folder containing `consolidated_metadata.csv` |
| `CLASSIFIER_MODEL_PATH` | `models/weights/baselineCNN.pth` | Path to CNN weights |
| `RL_MODEL_PATH` | `models/weights/baselineRL.pth` | Path to DQN weights |
| `LEGACY_CLOCK` | `True` | `True` = analog clock; `False` = digital countdown |

Action costs are defined in `gameplay/enums.py` as `ActionCost`:
`SAVE = 30`, `SQUISH = 5`, `SKIP = 15`, `SCRAM = 120`

---

## 15. Extension Points

The following are documented extension hooks intended for modification:

| Location | Hook | Common extensions |
|----------|------|-------------------|
| `gameplay/scorekeeper.py` `squish()` | `squished` counter incremented but not used in reward | Add a reward or penalty for squishing zombies |
| `gameplay/constants.py` — `HEADCOUNT_WEIGHT`, `VALUE_WEIGHT` | Linear combination weights for the reward signal — must sum to `1.0` (`VALUE_WEIGHT` is defined as `1.0 - HEADCOUNT_WEIGHT` so this is automatic). Default: both `0.5`. | Tune the balance between headcount and value signals; or modify `scorekeeper.get_cumulative_reward()` directly to replace the formula entirely |
| `game_interfaces/data_parser.py` `datarow_to_state()` | Module-level function, easily replaced | Customize how CSV rows map to humanoid states |
| `game_interfaces/data_parser.py` `datarow_to_value()` | Assigns `humanoid.value` — default is uniform random `[0.0, 1.0)` | Implement priority logic based on injury status, class, or other dataset signals |
| `gameplay/humanoid.py` `__init__` | `value` float set by `DataParser` via `datarow_to_value()` | `value` feeds into the reward and the RL observation space (slot `[14]`) |
| `gameplay/constants.py` | All constants exposed | Tune game parameters for experiments |
| `gameplay/constants.py` `LEGACY_CLOCK` | Feature flag for clock implementation | Swap UI components — use as a model for other feature flags |
| `ui/game_viewer.py` `display_score()` | Renders the end screen — currently shows `saved` and `killed` as plain labels | Add custom metrics, charts, or visualizations to the end screen |
| `ui/ui.py` `get_next()` | Determines the end reason and calls `game_viewer.display_score()` — controls what data is passed to the end screen | Add new end-screen metrics here before passing them to `display_score()` |
| `models/HumanoidClassifier.py`, `models/ActionRecommender.py` | Abstract base classes for classifier and recommender | Implement a new classifier or recommender from scratch |

> **Note:** The end-screen extension spans two files. `ui/game_viewer.py display_score()` owns the visual rendering; `ui/ui.py get_next()` owns the trigger logic and data selection. To add a new end-screen metric: (1) compute it in `get_next()`, (2) pass it to `display_score()`, (3) render it in `display_score()`.

---

### How to add a new humanoid class

Adding a new class (e.g. `HALF_ZOMBIE`) touches **5 files** in order. Missing any one causes a silent mismatch between training and inference.

1. **`gameplay/enums.py` — `State`**: add the new value to the enum. **Important:** the declaration order of `State` values is load-bearing — the obs vector encodes class counts in enum order. Always append new values at the end; inserting in the middle shifts all existing slot indices and silently breaks any trained model.
2. **`gameplay/scorekeeper.py` — `reset()`**: add a key for the new class in the `self.ambulance` dict (e.g. `"half_zombie": 0`). Also update any `save()` / `scram()` logic that branches on class.
3. **`gameplay/constants.py` — `OBS_SIZE`**: increment `OBS_SIZE` by 1 for each new class added. This is the single source of truth for the observation vector length — `obs_builder`, `train_env`, `RLRecommender`, and `DQNAgent` all read from it. If you skip this step the DQN input layer will have the wrong size and training will crash or silently use stale weights.
4. **`game_interfaces/obs_builder.py` — `build_obs()`**: add the new class to the `class_counts` block and update the `obs_idx["class_counts"]` slice in `base_env.__init__()` accordingly. This is the single source of truth for the observation encoding — both `TrainEnv` and `RLRecommender` delegate here, so one edit covers both training and inference.
5. **`training/train_rl.py`**: if you retrain after adding the class, the saved weights are now incompatible with the old `obs_size` — train from scratch or adjust the DQN input layer.

> **Why one file?** The obs encoding used to live in three places (`base_env._build_obs()`, `train_env._build_obs()`, `RLRecommender._build_obs()`), which had to be kept byte-for-byte identical by hand. They are now collapsed into `game_interfaces/obs_builder.py` — a single `build_obs()` function that both `TrainEnv` and `RLRecommender` delegate to. One change propagates automatically to both training and inference.
