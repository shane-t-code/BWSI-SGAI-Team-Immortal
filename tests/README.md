# Tests

Run all tests (with `bwsi-sgai` env active):

```
pytest
```

For VS Code setup and terminal usage see the top-level `README.md`.

---

## What each file covers

| File | What it tests |
|------|---------------|
| `test_data_model.py` | Enum ordering and `ActionCost` values — pins the contracts that the RL observation space and scoring depend on |
| `test_humanoid.py` | `Humanoid` construction, state predicate methods, and static index/string converters |
| `test_scorekeeper.py` | Scoring math — save, squish, skip, scram, capacity, available actions, cumulative reward, `get_score()` side effects |
| `test_data_parser.py` | CSV loading, `get_random()`, pool exhaustion, `reset()`, and `datarow_to_state()` / `datarow_to_value()` mapping |
| `test_cnn_classifier.py` | `CNNClassifier.classify()` — output shape, probability range, softmax sum; `probs_to_class()` label validity and confidence range |
| `test_oracle_classifier.py` | `OracleClassifier.classify()` — deterministic state-to-one-hot mapping for all four states; `probs_to_class()` round-trip |
| `test_heuristic_recommender.py` | `HeuristicRecommender.recommend()` — action selection covering all state × capacity combinations (note: `remaining_time=0` path tested via `test_scorekeeper.py`) |
| `test_rl_recommender.py` | `RLRecommender._build_obs()` — all 15 observation slots, range, one-hot encoding, time binning |
| `test_dqn_agent.py` | `DQNAgent` construction, `select_action()`, `train_step()`, `save()`/`load()` round-trip |
| `test_train_env.py` | `TrainEnv.step()` — blocked actions, terminal step, obs shape/range, reward delta |
| `test_build_obs.py` | `_build_obs()` sync — verifies `TrainEnv` and `RLRecommender` produce identical observations for the same inputs |
| `test_integration.py` | End-to-end episode loop — full game run with each classifier/recommender combination |
| `test_ui_assets.py` | UI asset presence — verifies logo and graphics files exist at expected paths |

## Shared fixtures (conftest.py)

`conftest.py` defines fixtures available to every test file. Run `pytest --fixtures`
to see full descriptions.

## Adding your own tests

1. Create a new file in `tests/` named `test_<your_thing>.py`
2. Write functions named `test_<what>_<expected_outcome>`
3. Use fixtures from `conftest.py` — or define your own in the same file
4. Run `pytest` — your tests are picked up automatically

Example:

```python
def test_scorekeeper_save_reduces_remaining_time():
    from gameplay.scorekeeper import ScoreKeeper
    from gameplay.humanoid import Humanoid
    sk = ScoreKeeper(shift_length=720, capacity=10)
    h = Humanoid(fp="x.jpg", state="healthy")
    sk.save(h)
    assert sk.remaining_time == 720 - 30  # SAVE costs 30 minutes
```
