# Data Logging Schema — Red/Blue Malice-Budget Experiment

Owner: Smit (data lead). This formalizes the trial-level schema so Shane's tier
generation, Claire's game-mode UI, and my analysis pipeline all agree on the
same field names and types. It is derived directly from the `recordAnswer()`
function already implemented in `index.html` (the checkpoint-terminal
prototype) — nothing here invents a new format, it documents and locks down
the one that already exists so it doesn't drift as the UI is wired up further.

## One row per trial (one humanoid judged by one participant)

| Field | Type | Source | Description |
|---|---|---|---|
| `participant_id` | string | `getOrCreateParticipantId()` in `index.html`, persisted via `localStorage` | Identifies one person across repeat plays on the same browser. See "Participant tracking" below. |
| `session_id` | string | generated fresh each page load (`crypto.randomUUID()`) | One sitting/playthrough — a participant who replays gets a new `session_id` each time but the same `participant_id`. |
| `game_set` | int (1-6) | `game_manifest.csv` / `rounds.json` `session` column | Which of Shane's 6 disjoint 30-round content sets was shown. **Not the same thing as `session_id` above** — `game_set` identifies the content shown, `session_id` identifies the sitting. Naming collision with the raw CSV's `session` column called out explicitly so it doesn't get confused with a participant's sitting. |
| `subject_id` | string | `currentRound.subjectId` (`game_manifest.csv` `subject_id`, e.g. `test_00074_tier1`) | Identifies which humanoid/image was shown. |
| `malice_tier` | int | `currentRound.tier` (`game_manifest.csv` `malice_tier`) | Independent variable. 4 confirmed levels: **`{0, 1, 5, 10}`** — updated by Shane in PR #7 after retraining with the two-channel (recolor + rotation) agent; supersedes the earlier `{0, 1, 3, 6}`. See "Tier definition" below. |
| `condition` | enum: `human_alone` \| `human_ai` | `getOrCreateParticipantId()` bootstrap in `index.html`, via `analysis/trial_protocol.assign_condition()` logic mirrored client-side | Assigned once per `participant_id` (between-subjects) — never changes across a participant's repeat sessions. See "Participant tracking" below. Still Claire's job to make SCAN/AI ASSIST actually respect this in `human_alone` (per her PR #7 assignment) — this field just records what was assigned. |
| `ground_truth_human` | bool | `currentRound.groundTruthIsHuman` | True if the subject is actually a healthy human, false if overgrown/zombie. |
| `player_answer_human` | bool \| null | argument to `recordAnswer()` | `null` = timed out, no decision made. |
| `correct` | bool \| null | computed: `player_answer_human === ground_truth_human` | `null` when `player_answer_human` is `null`. |
| `ai_confidence_shown` | float (0-100) | `currentRound.aiConfidence` (`game_manifest.csv` `ai_confidence_shown`, from the CNN's actual `cnn_p_healthy`) | The AI assistant's confidence — as of PR #7 this is the real classifier's output, not an invented number. |
| `reaction_time_ms` | int | `performance.now() - roundStartTime` | Primary dependent variable alongside `correct`. |
| `scanned` | bool | `scanned` toggle | Did the player open the Threat Advisory panel. |
| `zoomed` | bool | `zoomed` toggle | Did the player use zoom. |
| `ai_assist_used` | bool | `aiAssistUsed` toggle | Did the player open the AI panel this round. |
| `ai_assist_time_ms` | int \| null | `aiAssistClickTime` | Time into the round when AI assist was first opened; `null` if never opened. |
| `flagged` | bool | `flagged` toggle | Player marked the subject for review — not scored, just logged. |
| `timestamp` | ISO 8601 string | `new Date().toISOString()` | Wall-clock time the trial was recorded. |

## Data quality field (added by analysis, not the UI)

| Field | Type | Description |
|---|---|---|
| `flagged_invalid` | bool | Set by the analysis pipeline (see `analysis/data_quality.py`), not the game itself — `reaction_time_ms < MIN_VALID_RESPONSE_MS` (default 1000ms) or `player_answer_human is None`. Kept separate from the player-facing `flagged` field, which means something different (player suspicion, not data validity). |

## Tier definition — resolved twice now (Shane, PR #6 then PR #7)

Shane's `AdversarialEnv` (`game_interfaces/adversarial_env.py`) defines tiers as
edit-count snapshots along a single rollout. In PR #7 he retrained the agent
with a two-channel disguise (recolor for the human eye + rotation for the CNN)
and raised the budget so the top tier fools both channels, which changed the
snapshot points.

**Current confirmed tier set: `{0, 1, 5, 10}`** — `tier 0` is the real-human
control (no disguise); `1`/`5`/`10` are malice-budget edit-count snapshots.
This supersedes the `{0, 1, 3, 6}` set from PR #6. Verified directly against
`data/game_sets/game_manifest.csv` (180 rows, `malice_tier` column values
exactly `{0, 1, 5, 10}`).

Key finding from Shane's data (from his PR #7 summary, worth keeping in mind
for the write-up): the AI assistant's own accuracy drops sharply as tier rises
— correct 83% of the time at tier 1, down to 27% at tier 6-equivalent strength
in the old scheme (numbers superseded by the retrain, re-derive from
`game_manifest.csv`'s `cnn_correct` column once real trial data exists) —
meaning a human leaning on the assistant gets good advice against weak
disguises and actively bad advice against strong ones. That's the crux of the
`human_alone` vs `human_ai` comparison this whole pipeline exists to run.

## Participant tracking — resolved (Shane, PR #7)

Real playtesting (~30-40 people, many replaying) makes `participant_id` load-bearing:
without it, repeat plays from the same person are indistinguishable from new
participants, which would inflate the effective sample size and violate the
between-subjects design below.

**Scheme implemented in `index.html`:**
- `participant_id`: generated once via `crypto.randomUUID()` and persisted in
  `localStorage`. Same browser -> same `participant_id` across repeat visits.
  Known limitation: a participant who clears storage or switches devices/browsers
  looks like a new participant. Acceptable for an MVP playtest; not a real
  identity system.
- `condition`: assigned **once**, the first time a `participant_id` is created,
  and never changes on repeat visits (between-subjects, per Shane's note that
  AI advice teaches people what a disguise looks like — letting someone see
  both conditions would contaminate the comparison). Mirrors
  `analysis/trial_protocol.assign_condition()`'s alternation logic client-side
  (see the code comment in `index.html` for why this is duplicated rather than
  shared — no backend to be the single source of truth against).
- `session_id`: a fresh `crypto.randomUUID()` every page load/game start —
  identifies one sitting. A participant who plays 3 times has 1 `participant_id`,
  1 `condition`, and 3 distinct `session_id`s.

**Analysis implication:** the *primary* accuracy/response-time comparison uses
only each participant's first `session_id` (earliest `timestamp`) — pooling
every repeat session at the trial level would silently overweight whoever
replayed the most. Repeat sessions are kept in the raw data and are good
*exploratory* material (e.g. "does practice beat the disguise?") but are
excluded from the primary result. See `analysis/analyze_results.py`
`filter_first_session()`.

## Sample size — updated per Shane's numbers (PR #7)

With 15-20 participants per condition and 5 trials per malice tier per
participant: `15 x 5 = 75` to `20 x 5 = 100` trials per (tier, condition) cell
— comfortably clears the `>30` floor from `docs/trial-protocol.md`, about
2-3x over. This is the number to aim for when recruiting playtesters.

## Resolved from the original open questions list

1. ~~Condition assignment~~ — resolved above (between-subjects, assigned at
   first `participant_id` creation).
2. ~~`participant_id` / `session_id`~~ — resolved above.
3. ~~Where trial data is persisted~~ — `exportResultsCSV()` in `index.html`
   (added in PR #6) now includes real `participant_id`, `session_id`, and
   `condition` values instead of blanks/inferred placeholders.

## Still open

1. Claire still needs to wire `data/game_sets/rounds.json` into `index.html`
   in place of the placeholder `ROUNDS` array, and make SCAN/AI ASSIST
   actually respect the assigned `condition` (disabled entirely in
   `human_alone`) rather than always being available — her PR #7 assignment,
   not mine to patch unilaterally.
2. Prajwal still needs to fix `tests/test_scorekeeper.py::test_save_log_log_csv_columns`
   (missing `confidence_score` in `LOG_COLUMNS`) per Shane's note — unrelated
   to this pipeline, flagging here only because it's a known pre-existing
   failure noted twice now.
