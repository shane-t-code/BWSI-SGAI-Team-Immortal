# Instructor Guide — BWSI SGAI 2026

**For instructors running the codebase teaching block.**

This guide helps you facilitate the nine lecture notebooks and handle student questions during hands-on time. It assumes you have read the student docs (`docs/student-README.md`, `docs/architecture.md`) and are comfortable running the game yourself.

Student-facing troubleshooting reference: `docs/troubleshooting.md` — you can share this directly with students.

---

## Instructor Setup

Do this before the session. Students follow `docs/setup.md` — this section covers what the **instructor** needs to do to run the lectures from scratch.

### 1. Install prerequisites

- **Anaconda** — [anaconda.com/download](https://www.anaconda.com/download). Includes conda and Python. Install for "Just Me" (not all users) to avoid permission issues.
- **Git** — [git-scm.com/downloads](https://git-scm.com/downloads). On Windows: Git for Windows (includes Git Bash).
- **VS Code** — [code.visualstudio.com](https://code.visualstudio.com/). Install the Python extension after opening it.

### 2. Clone the repository

Open Anaconda Prompt (Windows) or Terminal (Mac/Linux):

```bash
git clone <repo-url>
cd bwsi-sgai
```

### 3. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate bwsi-sgai
```

This installs Python 3.13, PyTorch, Gymnasium, Tkinter, JupyterLab, and all dependencies. Takes 2–5 minutes. If `libmamba` is not set as the default solver on your machine and the install is slow, run:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Then retry `conda env create -f environment.yml`.

### 4. Register the conda env as a JupyterLab kernel

```bash
conda activate bwsi-sgai
python -m ipykernel install --user --name bwsi-sgai --display-name "BWSI SGAI"
```

This registers the conda env as a JupyterLab kernel. Without this step, notebooks will open with no kernel or the wrong Python.

### 5. Verify the game runs

```bash
python main.py -m human
```

A game window should open. If it doesn't, check `docs/troubleshooting.md` before the session.

### 6. Train the baseline models

The weight files are not in the repository. Train them once before the session:

```bash
python main.py train --type cnn     # ~2 minutes
python main.py train --type rl      # ~20–40 minutes
```

**Train the RL model the night before. Do not attempt it the morning of the session — it takes 20–40 minutes. Students will need the weights starting at Module 5.**

Keep a copy of the trained weights somewhere safe. If students overwrite them during experiments, you'll want the originals to restore from.

### 7. Open the lecture notebooks in JupyterLab

```bash
jupyter lab
```

This opens JupyterLab in your browser. Navigate to `lectures/` and open the first module notebook. The RISE extension turns the notebook into a slideshow — click the **Present** button (bar chart icon in the toolbar) to enter presentation mode.

**Keyboard shortcuts in RISE:**
- `Space` / `Shift+Space` — next / previous slide
- `F` — fullscreen
- `Esc` — exit slideshow, return to notebook
- `Shift+Enter` — run the current cell (works inside slideshow mode)

### 8. Verify the slideshow before the session

Run through each notebook in slideshow mode end-to-end. Run every code cell at least once to confirm they execute cleanly in your environment. Note any cells that take more than a few seconds — warn students before they run them.

---

## Overview

The codebase block is structured around nine interactive lecture modules (in `lectures/`), one per major directory. Students follow along on their own machines while the instructor drives on a shared screen. Each module covers one layer of the codebase, includes "change this and run" exercises, and ends with a pop quiz.

The goal is for students to leave with enough orientation to navigate the codebase independently and answer a research question with it.

| Module | Directory | Topic |
|--------|-----------|-------|
| 1 | `gameplay/` | Game mechanics, scoring, constants |
| 2 | `data/` | Dataset, metadata CSV, extension points |
| 3 | `ui/` | Tkinter UI, colors, clock widget |
| 4 | `game_interfaces/` + `main.py` | Architecture, wiring, OracleClassifier trace |
| 5 | `models/` (perception) | CNN, OracleClassifier, diagnostic swap |
| 6 | `models/` (decision) | HeuristicRecommender, RLRecommender, DQN |
| 7 | `training/` | Training loop, hyperparameters, reading results |
| 8 | `tests/` | pytest, coverage, writing assertions |
| 9 | Claude Code + `.md` docs | AI-assisted exploration, research question framing |

---

## Module Facilitation Notes

---

### Module 1 — Game Mechanics
**Notebook:** `lectures/module_01_game_mechanics.ipynb`

**What this module covers:** The four humanoid states, the four actions and their time costs (`ActionCost`), the `ScoreKeeper` reward formula, and every tunable constant in `gameplay/constants.py`. By the end, students have changed `HEADCOUNT_WEIGHT` and seen the formula react.

**The exercise:** Students open `gameplay/constants.py`, change `HEADCOUNT_WEIGHT` to `1.0`, restart the kernel, re-run the exercise cells, and compare the reward values. Then they try `0.0` and compare again. The notebook walks them through the formula step by step before they touch anything. The bonus step runs `python main.py -m auto` and reads `summary.csv` for a real game result. Allow 5–10 minutes for the exercise including the bonus.

**Common student questions:**

- *"Why does the score barely change when I set `HEADCOUNT_WEIGHT = 1.0`?"* — The simulated outcome in the notebook is short (4 saves, 1 kill). The absolute difference is small, but the ranking between scenarios does change. The effect is more visible in a full auto game via `summary.csv`.
- *"Does changing the constant retrain the agent?"* — No. The RL agent was trained with `HEADCOUNT_WEIGHT = 0.5`. Changing it in `constants.py` changes how the current game is *evaluated*, not how the agent decides. To test a new weight during training, change it and retrain.
- *"What does `seed=42` do in `ScoreKeeper`?"* — It records the seed in `summary.csv` for reproducibility. It does NOT control which humanoids appear — that's `DataParser`'s seed, set separately.

**Mid-lecture recovery:** If students can't run the exercise cells (import error, wrong kernel), check that JupyterLab was launched from the repo root and the kernel is set to `bwsi-sgai`. Don't spend more than 2 minutes debugging live — have them watch you run it on the projector and catch up at the bonus step.

---

### Module 2 — Data
**Notebook:** `lectures/module_02_data.ipynb`

**What this module covers:** The 200-image dataset, `consolidated_metadata.csv`, the five CSV columns (`Class`, `Injured`, `Gender`, `Item`, `Filename`), how those columns are used to derive the four game states (via `datarow_to_state()`), and the `Item`/`Gender` columns as extension points. Students also see the class imbalance (healthy=81, zombie=46, corpse=44, injured=29) and understand why it matters for CNN training.

Note: the CSV has no `State` column. The state numbers are computed by `datarow_to_state()` at runtime. The `Class` column holds `Default`/`non-Default` (not the state label), and `Injured` is a bool.

**The exercise:** Pure pandas — no game loop, no weights needed. Students filter the DataFrame to find all injured humanoids: `df[df['Injured'] == True]` (answer: 29 rows). Then they inspect the `Item` column on those rows. They write their own filter (female zombies: `df[(df['Gender'] == 'Female') & (df['Class'] == 'non-Default')]`, Bow carriers, etc.). Allow 5 minutes. Students who finish early: write the `datarow_to_value()` rule for Bow-carriers and test it with `df.iloc[6]`.

**Common student questions:**

- *"Where does the CSV come from? Were these images generated?"* — The images are real game assets, hand-labeled. The CSV is the ground truth — it was produced manually, not by the model.
- *"Why doesn't my filter work when I write `== 'True'` for Injured?"* — `Injured` is a bool dtype, not a string. Use `== True` (Python boolean), not `== 'True'`. This catches everyone at least once.
- *"What's the NaN in the Item column?"* — Humanoids with no item get a literal pandas `NaN`, not the string `'NaN'`. Check for it with `pd.isna(row["Item"])`, not `== "NaN"`.

**Mid-lecture recovery:** If pandas fails to load the CSV, the most likely cause is that JupyterLab is not running from the repo root — the relative path `data/consolidated_metadata.csv` won't resolve. Ask students to run `import os; print(os.getcwd())` in a cell. If it shows a `lectures/` subdirectory, they need to restart JupyterLab from the repo root.

---

### Module 3 — User Interface
**Notebook:** `lectures/module_03_ui.ipynb`

**What this module covers:** The Tkinter UI layout, `ui.py` and `button_menu.py`, the `UI_COLORS` dict in `constants.py`, the digital clock (`digital_clock.py`) and capacity meter (`capacity_meter.py`) widgets, and the `LEGACY_CLOCK` toggle.

**The exercise:** Students open `gameplay/constants.py`, change one color in `UI_COLORS` (e.g. the background), save the file, and relaunch `python main.py -m human` in a terminal to see the change. A second optional step: toggle `LEGACY_CLOCK = False` and relaunch to switch from the default analog clock to the seven-segment digital display. Allow 5–8 minutes including the relaunch.

**Common student questions:**

- *"Can I run the UI inside the notebook?"* — No. Tkinter requires a real display and event loop. Running `python main.py -m human` from the notebook would block the kernel and never return. They must use a terminal alongside the notebook.
- *"The window didn't change — did I save the file?"* — Almost always a save issue. Confirm `Ctrl+S` in VS Code before relaunching. Also check they changed `UI_COLORS` in `gameplay/constants.py`, not a different file.
- *"What's LEGACY_CLOCK?"* — It switches between the analog elapsed-time clock (`clock.py`) and the digital countdown clock (`digital_clock.py`). The default is `True` (analog). Setting it to `False` gives the seven-segment digital display. It's cosmetic only.

**Mid-lecture recovery:** If students can't open a terminal window alongside the notebook, have them use the JupyterLab built-in terminal (File → New → Terminal). It opens in the repo root, which is exactly what's needed. If the game window opens but immediately crashes, check `docs/troubleshooting.md` for the Tkinter/DLL issues.

---

### Module 4 — Architecture and Wiring
**Notebook:** `lectures/module_04_architecture.ipynb`

**What this module covers:** How `main.py` wires `DataParser`, classifier, recommender, and `ScoreKeeper` together. The base game loop in `game_interfaces/base_env.py`. The observation vector (15 slots). Tracing a full turn using `OracleClassifier` so no weights are needed.

**The exercise:** Students trace a full turn by adding a `print()` statement inside `base_env.py`'s game loop, running `python main.py -m auto` for a few turns, and reading the printed output. No file is copied; no weights needed because the exercise uses `OracleClassifier`. Allow 5–8 minutes.

**Common student questions:**

- *"Why would you ever use Oracle in a real game?"* — You wouldn't deploy it, but you use it to isolate layers during debugging. If the game plays well with Oracle but poorly with CNN, the CNN is the problem. If it plays poorly with both, the RL policy is the problem. This is covered more in Module 5.
- *"What's an abstract base class?"* — Refer them to Module 5 (ABCs) where it's covered properly. For now: it's a shared contract that both `CNNClassifier` and `OracleClassifier` follow, so the game loop can swap them without knowing which one it has.
- *"Where is the game loop — what calls what?"* — The call chain is: `main.py → Main._run_auto() → DataParser.get_random() → classifier.classify() → recommender.recommend() → scorekeeper.save/skip/squish/scram()`. `_run_auto()` is a method of `Main` (in `main.py`), not `BaseEnv` — `BaseEnv` handles the GUI game modes. Tracing this on the whiteboard before the exercise is worth 2 minutes.

**Mid-lecture recovery:** If the auto game crashes mid-trace (common if the `print()` is placed incorrectly), show students how to use `git restore game_interfaces/base_env.py` to revert. Then walk through the correct placement together before they try again.

---

### Module 5 — Perception / CNN
**Notebook:** `lectures/module_05_perception.ipynb`

**What this module covers:** The `HumanoidClassifier` ABC and its one abstract method `classify()`, `probs_to_class()` as the correct way to convert probabilities to a label, `CNNClassifier` construction and the `FileNotFoundError` fail-fast pattern, `OracleClassifier` as a diagnostic tool, and `datarow_to_value()` as the triage value extension point.

**The exercise:** Students add an `OracleClassifier` import to `main.py`, search `main.py` for all occurrences of `CNNClassifier` (there are 5: 4 in `_make_components()` across mode branches, 1 in `_run_auto()`), and replace all 5 with `OracleClassifier()`. For a minimal test, replacing only the one in `_run_auto()` is enough to see oracle behavior in `auto` mode. Run `python main.py -m auto --log --seed 42` and compare the reward to the CNN baseline in `summary.csv`. Then revert with `git restore main.py`. Allow 5–10 minutes including the revert step.

**Common student questions:**

- *"Why does `classify()` take a `Humanoid` object instead of just an image path?"* — `CNNClassifier.classify()` only uses `humanoid.fp`. `OracleClassifier.classify()` only uses `humanoid.state`. Both satisfy the same interface because they take the same argument — the caller doesn't need to know which one it has.
- *"Why is Oracle confidence always 1.0?"* — Because it returns a one-hot vector — all probability mass on one class, zero on the others. `probs_to_class()` returns `float(probs[idx])` where `idx` is the argmax; for a one-hot vector, that value is 1.0.
- *"Can I write my own classifier?"* — Yes. Inherit from `HumanoidClassifier`, implement `classify(humanoid) -> np.ndarray`, return a `(4,)` float32 array in `[zombie, healthy, injured, corpse]` order. Search `main.py` for `CNNClassifier` — there are 5 occurrences (4 in `_make_components()`, 1 in `_run_auto()`) — and replace whichever modes you want to test.

**Mid-lecture recovery:** There are 5 occurrences of `CNNClassifier` in `main.py` — 4 in `_make_components()` (one per mode branch) and 1 in `_run_auto()`. If students only change the `_run_auto()` occurrence, GUI modes will still use CNN while `auto` mode uses Oracle. Walk the class through `Ctrl+F` for `CNNClassifier` in `main.py` before they start editing so every occurrence is visible. If a student's swap breaks the run, `git restore main.py` resets it cleanly.

---

### Module 6 — Decision / RL
**Notebook:** `lectures/module_06_decision.ipynb`

**What this module covers:** The `ActionRecommender` ABC and the `recommend(humanoid, probs, scorekeeper)` contract, `HeuristicRecommender` as the lowest-friction decision entry point (plain if/elif rules), `RLRecommender` as the trained DQN wrapper, and `DQN.py` / `DQNAgent.py` as the underlying network and training harness (for context, not editing).

**The exercise:** Students activate `HeuristicRecommender` in `main.py`, open `HeuristicRecommender.py` and change one rule (add a Skip case, change the Squish threshold, or invent their own), run `python main.py -m auto --log --seed 42`, and compare action counts in `log.csv`. Then they revert `main.py`. Allow 8–12 minutes — this exercise has the most steps of any module.

**Common student questions:**

- *"Why isn't HeuristicRecommender active by default?"* — `RLRecommender` is the production default. `HeuristicRecommender` is a research surface. Students activate it explicitly because they need to understand that the choice of recommender is a deliberate wiring decision in `main.py`, not automatic.
- *"My rule returned `'save'` as a string and the game crashed — why?"* — `recommend()` must return `ActionState.SAVE`, not the string `'save'`. Show the `ActionState` enum and the `TypeError` message so they recognize it next time.
- *"How does the RL agent actually decide?"* — It builds a 15-slot observation vector from game state + probs + available actions, runs a forward pass through the DQN, and picks `argmax` of the four Q-values. The training is in Module 7.

**Mid-lecture recovery:** This module has the most common failure point in the codebase: students forget to activate `HeuristicRecommender` in `main.py` before running. If `python main.py -m auto` is using the RL weights and students are confused why their rule change has no effect, check `main.py` first — `RLRecommender()` is almost certainly still there. Walk through the activation step explicitly before anyone opens `HeuristicRecommender.py`: edit `main.py` to replace `RLRecommender(...)` with `HeuristicRecommender()` in both `_make_components()` and `_run_auto()`.

---

### Module 7 — Training
**Notebook:** `lectures/module_07_training.ipynb`

**What this module covers:** `training/train_rl.py` and `training/train_cnn.py`, how to read `summary.csv` after a training run, key RL hyperparameters (`NUM_EPISODES`, `EPSILON_DECAY`, `GAMMA`), CNN hyperparameters (`NUM_EPOCHS`, augmentation), and when weights are saved during RL training.

**The exercise:** Students run `python main.py train --type rl -e 500` in the terminal, then read `summary.csv` to find the final reward. They note the episode count and compare to the pre-trained 2500-episode baseline. Allow 10–15 minutes for the run itself (500 episodes takes 5–10 minutes on CPU). While the run is live, walk through the training loop code in the notebook.

**Common student questions:**

- *"I ran 500 episodes but the weights file didn't change — why?"* — RL weights only save when `episode_reward > best_reward AND epsilon < 0.5`. If the 500-episode policy never beat the initial baseline, no save occurs. Check the terminal output — it prints a message when weights are saved. If nothing better was found, that is the correct behavior: the agent didn't improve, so the old weights are preserved.
- *"Should I use the `.ipynb` or the `.py` file to set hyperparameters?"* — Pick one interface and stay consistent. The canonical defaults are in the `.py` file. If you've changed both, compare the hyperparameter blocks manually — they're separate copies.
- *"Why is CNN training capped at 20 epochs?"* — The dataset has 160 training images (80/20 split from 200). Overfitting is the dominant risk past 20 epochs. The notebook includes an overfitting warning — if validation accuracy stops improving while training accuracy keeps climbing, stop early.

**Mid-lecture recovery:** If a student's training run stalls or crashes partway through, check that the conda env is active (`conda activate bwsi-sgai`) and that they're running from the repo root. On Windows, RL training can occasionally hang on BLAS DLL issues — if so, add the env's `Library\bin` to PATH per `docs/windows-setup.md`. For a crashed run, existing weights are unaffected — the save only happens when a better policy is found, and it's atomic.

If a student's 500-episode run hasn't finished after 15 minutes, have them kill the process (`Ctrl+C`) and instead read the pre-trained weights: `python main.py -m auto --log --seed 42` uses whatever weights are currently in `models/weights/baselineRL.pth`. They can compare that result against scores from their own run when it completes.

If students see no output for several minutes during RL training, Python may be buffering stdout. Rerun with `python -u main.py train --type rl -e 500` for unbuffered output.

---

### Module 8 — Tests
**Notebook:** `lectures/module_08_tests.ipynb`

**What this module covers:** The `tests/` directory structure, how to run `pytest` and `pytest --cov`, reading a coverage report, the difference between unit and integration tests, and how to add one assertion to an existing test.

**The exercise:** Students run `pytest tests/ --cov` from the terminal, read the coverage report, and then open an existing test file and add one new assertion to an existing test function (not a new test — just an assertion). They run pytest again and confirm the new assertion passes. Allow 5–8 minutes.

**Common student questions:**

- *"Can I run pytest from inside the notebook?"* — It works but output is messy. Recommend the terminal. The command is the same.
- *"What does 81% coverage mean?"* — 81% of the codebase lines are executed during the test suite. `main.py` has lower coverage (~45%) because some modes require user interaction. That's acceptable.
- *"Do I need to understand every test before adding an assertion?"* — No. Find a test for something they just learned — e.g. `test_scorekeeper.py` or `test_data_parser.py` — add `assert result is not None` or a similar sanity check. The goal is confidence with the workflow, not test design mastery.

**Mid-lecture recovery:** The most common failure is running `pytest` with the wrong conda env active. If imports fail inside pytest, `conda activate bwsi-sgai` first, then retry. If students see all tests passing on a clean run, the environment is correct. If they see failures, don't debug live — confirm the test count matches and move on.

---

### Module 9 — AI-Assisted Exploration
**Notebook:** `lectures/module_09_ai_exploration.ipynb`

**What this module covers:** How to use Claude Code (or another LLM with the module reference docs) to navigate and extend the codebase. Students practice three prompt types: explain a function, suggest an experiment, and debug an error. This is also the module where teams draft their research question.

**The exercise:** Students write three prompts targeting a specific file and question. Example sequence: (1) "Explain what `datarow_to_value()` does and what columns it can use" — (2) "I want to prioritize injured humanoids in `datarow_to_value()`. Show me the code change and which files I'd need to edit" — (3) paste an error message and ask for the fix. After the three prompts, teams spend 5–10 minutes drafting a research question. Allow 15–20 minutes total.

**Common student questions:**

- *"What makes a good prompt?"* — Specific file + specific question = useful answer. "What does `HeuristicRecommender` do" is vague. "What does `recommend()` in `models/HeuristicRecommender.py` return when the ambulance is full, and what line controls that?" is specific.
- *"Can the AI just write the code for us?"* — It can suggest it. The requirement is that your team understands the change well enough to explain it in the presentation. AI-generated code you can't explain is a presentation liability.
- *"How do we know our research question is answerable?"* — See Section 4 (Research Question Arc) below. Bring up the criteria in class and have each team check their question against them before the session ends.

**Mid-lecture recovery:** If token limits or access issues prevent live AI use, pivot to the module reference docs in each directory (e.g. `models/models.md`, `gameplay/gameplay.md`). These have the same structure — Mermaid diagrams, per-file tables, extension points. Teams can draft their research question from the docs alone. The research question draft is the deliverable; the AI prompting is the method.

---

## Research Question Arc

Teams need to go from "we changed something" to "we have a finding to present." This section covers how to guide that transition.

### What makes a measurable question

A good research question has three components:
- A **specific variable** that you change (one thing, defined precisely)
- A **specific metric** that you measure (reward, kill count, save count, action distribution)
- A **specific comparison** (condition A vs. condition B, both run the same way)

### Examples

**Good:** "Does setting `GAMMA = 0.9` vs. `GAMMA = 0.99` affect the final reward in `auto` mode after 500 training episodes, with seed 42?"

- Variable: `GAMMA` (one constant, two values)
- Metric: `reward` column in `summary.csv`
- Comparison: both conditions trained for exactly 500 episodes, same seed

**Bad:** "Does the game get better if we train longer?"

- "Better" is undefined
- "Longer" is undefined
- No fixed comparison — are you comparing 500 vs. 2500? What stays constant?

**Good:** "Does `HeuristicRecommender` with a value-based Skip rule produce more `skip` actions than the default rule, and does that affect final score?"

- Variable: the Skip rule in `HeuristicRecommender.py`
- Metric: action counts in `log.csv`, reward in `summary.csv`
- Comparison: default vs. modified rule, same seed, same episode count (if training is involved)

**Bad:** "Is `HeuristicRecommender` better than `RLRecommender`?"

- "Better" is undefined
- No control for episode count, seed, or metric

### The comparison trap

If one condition uses more training episodes than another, the team is measuring training time, not their experimental variable. Both conditions must use the same `NUM_EPISODES`. This applies to every comparison involving the RL agent.

**Heuristic vs. RL is not a controlled experiment unless both are evaluated under the same game configuration the RL agent was trained on.** If a team changes `HEADCOUNT_WEIGHT` after training, the RL policy is miscalibrated for the new reward signal — the comparison is invalid. Either retrain the RL agent after any constant change, or only compare agents that were trained under the same constants.

### How to read results

Run `auto` mode with `--log --seed 42`:

```bash
python main.py -m auto --log --seed 42
```

This writes to `summary.csv`. Read the `reward` column. For more reliability, run 3–5 seeds and average:

```bash
for seed in 42 0 1 7 99; do
    python main.py -m auto --log --seed $seed
done
python -c "import pandas as pd; df = pd.read_csv('summary.csv'); print(df.tail(5)[['seed','reward','saved','killed']].to_string())"
```

Delete `log.csv` and `summary.csv` before starting a new experiment condition — they append across runs.

### Timeline guidance

| Point in the session | What teams should have |
|----------------------|----------------------|
| End of Module 9 | A drafted research question (checked against the criteria above) |
| Next session | Experiment data collected (both conditions run) |
| Final session | Results interpreted; presentation drafted |

Teams that don't lock a question by end of Module 9 tend to run out of time for data collection. Push them to commit to a question — even an imperfect one is better than no question.

---

## Extension Points

When students ask "what can I change?", point them here. These are the safe experiment surfaces — changes that don't require understanding infrastructure code.

### `gameplay/constants.py` — easiest starting point

| Constant | Default | What changing it does | Retrain? |
|---|---|---|---|
| `HEADCOUNT_WEIGHT` | `0.5` | Shifts reward toward headcount vs. value | RL yes |
| `SHIFT_LENGTH` | `720` | Changes game length. Note: shorter shifts make SCRAM (120 min) proportionally more expensive — shorter ≠ more scramming | RL recommended |
| `AMBULANCE_CAPACITY` | `10` | How many humanoids fit on ambulance | RL recommended |
| `UI_COLORS` | dict | Cosmetic only | No |

### `gameplay/enums.py` — action costs

Changing `ActionCost` values (e.g., making `SQUISH` cheaper) changes the trade-off the RL agent faces. Retrain after any change. Warn students that `ActionState` and `ActionCost` must stay in sync — adding to one without the other produces a **silent no-op** (not an error) in `map_do_action()`.

### `game_interfaces/data_parser.py` — triage value

`datarow_to_value()` is the most accessible extension. Default returns a random float. Students can make it depend on `Injured`, `Gender`, or `Item` columns from the metadata CSV. Return value must stay in `[0.0, 1.0]`.

The `rng` parameter is the seeded `random.Random` instance — use `rng.random()` instead of `random.random()` if the value should be random but reproducible with `--seed`.

### `models/HeuristicRecommender.py` — rule-based strategy, no training needed

The simplest way to change decision-making. No weights, no retraining. **No CLI mode uses `HeuristicRecommender` by default** — students must edit `main.py` directly to activate it (replace `RLRecommender(...)` with `HeuristicRecommender()` in `_make_components()` and `_run_auto()`). This is the most commonly missed step in Module 6.

### `training/train_rl.py` and `training/train_cnn.py` — hyperparameters

RL: `NUM_EPISODES` (2500 default; 500 is a reasonable fast baseline), `EPSILON_DECAY`, `GAMMA`. CNN: `NUM_EPOCHS` (stay under 20 on a 160-image dataset — overfitting risk).

**Comparison trap:** If a team compares a 500-episode agent against the 2500-episode baseline, they're comparing training time, not their experimental variable. Both conditions should use the same episode count.

---

## Facilitation — General Moves

These apply across all modules regardless of notebook content.

**When a student can't launch the game mid-lecture:**
"Does `python main.py -m human` open a window?" If no — check conda env first (`conda activate bwsi-sgai`), then check `docs/troubleshooting.md`. Don't troubleshoot live for more than 2 minutes if others are waiting. Have the student follow along by watching and catch up at the next natural pause.

**When many students hit the same error simultaneously:**
Stop the lecture. Fix it together as a class — this is more efficient than parallel individual debugging and the fix is usually the same for everyone (wrong env, missing weights, PATH issue).

**When a student is confused about where something lives:**
Point them to the module reference doc for that directory (e.g., `gameplay/gameplay.md`). These have Mermaid diagrams and per-file tables. The pattern: find the file first, then the function.

**Diagnosing CNN vs. RL problems:**
The `OracleClassifier` swap is the most useful single diagnostic move in the codebase. See `docs/troubleshooting.md` for swap instructions. If the game plays well with oracle but poorly with CNN, the CNN is the bottleneck. If it plays poorly with both, the RL policy needs work.

---

## Quick Reference

```bash
conda activate bwsi-sgai
python main.py -m human                      # human mode, no weights needed
python main.py -m auto --log --seed 42       # headless RL, logging on, seeded
python main.py train --type cnn              # train CNN (~2 min on CPU)
python main.py train --type rl               # train RL, 2500 episodes (~20–40 min)
python main.py train --type rl -e 500        # fast RL baseline (~5–10 min)
python -u main.py train --type rl -e 500     # same, unbuffered stdout (use if no output appears)
python -m pytest tests/                      # run all tests
```

### Game modes

| Mode | What it does | Needs CNN | Needs RL |
|------|-------------|-----------|----------|
| `human` | Human plays, no AI | No | No |
| `human_cnn` | CNN labels humanoids, human acts | Yes | No |
| `human_rl` | RL recommends actions, human acts | Yes | Yes |
| `human_cnn_rl` | CNN labels + RL recommends, human acts | Yes | Yes |
| `machine` | CNN + RL play; press button to advance | Yes | Yes |
| `auto` | Headless RL run, logs score | Yes | Yes |

### Key files

| File | What to change |
|------|----------------|
| `gameplay/constants.py` | `HEADCOUNT_WEIGHT`, `SHIFT_LENGTH`, `AMBULANCE_CAPACITY` |
| `gameplay/enums.py` | `ActionCost` values (keep in sync with `ActionState`) |
| `game_interfaces/data_parser.py` | `datarow_to_value()`, `datarow_to_state()` |
| `models/HeuristicRecommender.py` | Decision rules (activate by editing `main.py` first) |
| `training/train_rl.py` | `NUM_EPISODES`, `GAMMA`, `EPSILON_DECAY` |
| `training/train_cnn.py` | `NUM_EPOCHS` (stay under 20) |

---

## End-of-Day Cooldown Activity

Optional ~20-minute activity for after the last module. Three segments: speed run (teams race to find things in the code), prediction game (live constant changes with team predictions), and hot take debate (design decisions argued by teams). No prep required.

See `docs/cooldown-activity.md` for the full instructor guide.

---

## Known uncertainties in this guide

The following items are based on the notebooks as written but have not been verified against a live student session:

- **Module 3 (UI) exercise timing:** Relaunching `python main.py -m human` to see a color change adds latency (Tkinter startup, especially on Windows). The "5–8 minutes" estimate may be tight if students are also dealing with terminal setup for the first time.
- **Module 7 (Training) recovery for stalled runs:** The guidance about BLAS DLL issues causing hangs is based on known Windows behavior with numpy, not a confirmed symptom of RL training specifically. If a hang occurs and PATH is already set correctly, check whether PyTorch is using CPU vs. GPU — a missing CUDA fallback can cause similar symptoms.
- **Module 9 (AI-Assisted) token access:** This guide assumes students have access to Claude Code or a comparable LLM during the session. If access is unavailable, the module reference docs (`gameplay/gameplay.md`, `models/models.md`, etc.) are the fallback and cover the same ground.
- **`CNNClassifier` occurrence count (Module 5, Module 6):** The guide states there are 5 occurrences of `CNNClassifier` in `main.py` — 4 in `_make_components()` (one per mode branch) and 1 in `_run_auto()`. If the codebase is refactored to consolidate these, update the facilitation notes for both modules.
