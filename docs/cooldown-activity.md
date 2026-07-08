# End-of-Day Cooldown — Codebase Surprises

**For instructors.** Run this after the last code lecture module, before teams break. Takes about 20 minutes. Students don't have to play — participation is completely optional. The activity works fine with 2 engaged teams and a room of tired people half-listening.

No prep required beyond having the repo open.

---

## The Arc

Three segments in order:

1. **Speed run** (~5 min) — teams race to find things in the code
2. **Prediction game** (~10 min) — you make a live change, they predict what happens
3. **Hot take** (~5 min) — you read a design decision, teams argue about it

Run them together or pick one. Stop whenever the energy drops.

---

## Segment 1 — Speed Run

Teams compete. First team to find the answer and call it out wins the round. They should have their laptops open to the repo.

Read each challenge out loud. Don't explain it — the weirdness is the point.

---

**Round 1**
> "Find the function in this codebase whose entire body is two lines ending with: `return rng.random()`. What file is it in, and what is it supposedly doing?"

*Answer:* `datarow_to_value()` in `game_interfaces/data_parser.py`. It's the triage priority function. Every humanoid's priority is a random float. That's it. That's the whole function.

---

**Round 2**
> "There's one action in this game that can never be blocked — no matter what's happening, the game will always let you do it. Find the comment that explicitly says so."

*Answer:* `scorekeeper.py`, line 354: `# scram is always available — never disabled per design doc §8`

---

**Round 3**
> "Somewhere in this codebase there's a comment that cites a research paper from 2015. Find it."

*Answer:* `models/DQNAgent.py`, the `SmoothL1Loss` line. The comment ends with "This matches the original DQN paper (Mnih 2015)."

---

**Round 4**
> "Find the place where killing someone is exactly three times faster than leaving them behind."

*Answer:* `gameplay/enums.py` — `ActionCost.SQUISH = 5`, `ActionCost.SKIP = 15`. The docstring explains why.

---

## Segment 2 — Prediction Game

You drive. Teams predict. Don't reveal the answer until after they've committed.

Have `gameplay/constants.py` open and `auto` mode ready to run:

```bash
python main.py -m auto --log --seed 42
```

Run it once before making any changes so teams have a baseline in their head (or just tell them the baseline reward from the pre-trained run).

---

**Round 1**

Show teams this line in `constants.py`:

```python
HEADCOUNT_WEIGHT: float = 0.5
VALUE_WEIGHT: float = 1.0 - HEADCOUNT_WEIGHT
```

Ask: *"If I change `HEADCOUNT_WEIGHT` to `1.0` — what does the agent start caring about? What does it stop caring about?"*

Wait for predictions. Then make the change and run `auto` mode. Watch the score.

*What actually happens:* The agent now optimizes purely for `saved − killed`. The `VALUE_WEIGHT` drops to `0.0`, so humanoid priority values disappear entirely from the reward signal. An injured humanoid with value 0.1 is now worth exactly as much as one with value 0.9. The agent doesn't know the difference.

---

**Round 2**

Change `HEADCOUNT_WEIGHT` to `0.0`. Ask: *"Now the agent doesn't care at all whether people live or die. It only cares about value. What do you think it does differently?"*

*What actually happens:* The agent optimizes entirely for `saved_value`. Saving a low-value humanoid is almost worthless. The total reward can still look reasonable if it's lucky about who shows up — or it can collapse if high-value humanoids don't appear. The kill count might not change much; what changes is *why* the agent makes the decisions it does.

*Good follow-up question for the room:* "Which version of the agent would you want running in a real triage situation?"

---

**Reset before leaving this segment:** Restore `HEADCOUNT_WEIGHT = 0.5` before students start their own experiments tomorrow.

---

## Segment 3 — Hot Take

Pick one. Read it out loud. Give each team 60 seconds to decide which side they're on, then let them argue. No right answer — the point is that there's something real to argue about.

---

**Hot Take A — The Time Problem**

> "SKIP costs 15 minutes. SQUISH costs 5. In this game, killing someone is three times faster than leaving them behind. That's a deliberate design choice. Is it good design — or is it a moral failure baked into the rules?"

*Background if they need it:* The `ActionCost` docstring explains the reasoning: skip means navigating around the humanoid and leaving them there, which takes longer than a quick squish. The asymmetry matters for the RL agent because squishing is always faster, even when it kills. The agent has to learn to weigh speed against humanitarian outcomes.

---

**Hot Take B — The Random Priority Problem**

> "Every humanoid's triage priority in this game is assigned randomly at game start. You can't tell from the image who has a high value or a low value. The code literally says `return rng.random()`. Is random fair — or is it a cop-out that avoids a harder design problem?"

*Background if they need it:* `datarow_to_value()` in `data_parser.py` is the function. The metadata CSV has columns for `Gender`, `Item`, and `Injured` — any of these could be used to assign value in a non-random way. The docstring explicitly invites students to replace the random assignment with something principled. The question is whether "principled" is better, or whether it just smuggles in a different set of assumptions.

---

## Notes

- Teams that just watched without playing often have the best hot take opinions — don't skip them.
- If you're running short on time, Speed Run alone works as a standalone activity.
- The prediction game requires you to have `auto` mode running cleanly before the session. Test it once before you need it.
