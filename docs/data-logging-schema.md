# Data Logging Schema — Red/Blue Malice-Budget Experiment

Owner: Smit (data lead). This formalizes the trial-level schema so Shane's tier
generation, Claire's game-mode UI, and my analysis pipeline all agree on the
same field names and types. It is derived directly from the `recordAnswer()`
function already implemented in `index.html` (the checkpoint-terminal
prototype) — nothing here invents a new format, it documents and locks down
the one that already exists so it doesn't drift as the UI is wired up further.

**Update (Claire, `screen.html`):** the SCAN, AI ASSIST, and FLAG buttons
were removed from the UI. The Threat Advisory and AI Assistant panels are no
longer opened by a click — they now render automatically for every
`human_ai`-condition round (and never for `human_alone`), which is a cleaner
way of enforcing the "disabled entirely in `human_alone`" requirement below
than gating individual buttons was. `condition` alone now fully determines
whether the panels were shown, so `scanned`, `ai_assist_used`, and
`ai_assist_time_ms` were dropped from the schema — they had become
dead/derived fields once the buttons were removed. `flagged` was also dropped
— it had no button behind it and was always `false`. The old boolean `zoomed`
field was replaced by `zoom_level` (int 0-3) since zoom is now 3 discrete
magnification settings (ZOOM 1/2/3) instead of a single on/off toggle. See the
field table below for the updated per-field detail.

**Update (2026-07-25):** `player_answer_human`'s values changed from
`"human"`/`"zombie"` to `"permit"`/`"reject"` — it now logs the player's
literal checkpoint action rather than translating it into a class judgment;
`ground_truth_human` and `cnn_says_human` are unaffected and still log
classes.

An earlier version of this update also gave `ground_truth_human` a third
value, `"dead"`, for the tier-0 humans the CNN itself classifies as a corpse
(`cnn_predicted == "corpse"`), and excluded those trials from `correct`
scoring. That was reverted the same day: those subjects are the *injured*
humans in the pool (`consolidated_metadata.csv`'s `Injured` flag), and
`true_class == "human"` and `cnn_predicted == "corpse"` are exactly the ones
the classifier got wrong — so `"dead"` was deriving ground truth from the
CNN's own error, meaning two physically identical injured subjects could get
different ground truths depending only on whether the CNN happened to
misclassify that particular render. It also made those rows unscoreable,
discarding the clearest over-trust case in the dataset ("AI confidently
called a savable person dead — did the player follow?"). Ground truth is
back to a plain `"human"` \| `"zombie"` boolean-backed label; wounded-but-alive
is still PERMIT. The distinction is preserved separately as the descriptive
`subject_injured` column in `game_manifest.csv` (sourced from
`consolidated_metadata.csv`, joined by `curate_game_sets.py`'s
`load_injury_lookup()`) — it doesn't affect `correct` scoring, but supports
analysis like "do players reject injured humans more often?".

## One row per trial (one humanoid judged by one participant)

| Field | Type | Source | Description |
|---|---|---|---|
| `participant_id` | string | `getOrCreateParticipantId()` in `screen.html`, persisted via `localStorage` | Identifies one person across repeat plays on the same browser. See "Participant tracking" below. |
| `prev_participant_id` | string \| null | last entry of the `participant_id_chain` in `localStorage` | The id this browser used *before* the most recent dev-reset, so a reset-and-replay browser is traceable ("played before as X, came back as Y"). `null` if this browser was never reset. |
| `session_id` | string | generated fresh each page load (`crypto.randomUUID()`) | One sitting/playthrough — a participant who replays gets a new `session_id` each time but the same `participant_id`. |
| `content_session_key` | string `"1"`–`"6"` | `getNextContentSession()` — rotates `play_count % 6` per page load | Which of the 6 disjoint 30-round content sets (`rounds.json` keys) was dealt. This is the field the earlier draft called `game_set`. **Not the same thing as `session_id` above** — it identifies the content shown, `session_id` identifies the sitting. |
| `play_count` | int | `session_count_<participant_id>` counter in `localStorage` | How many page loads this participant had before this one (`0` = first sitting). Increments on page load, not on game completion, so an abandoned/refreshed load still advances the content rotation. |
| `station` | string | `station.meshName` | Which in-world screen the trial was answered at (currently only `screenMain` exists in the map). |
| `subject_id` | string | `currentRound.subjectId` (`game_manifest.csv` `subject_id`, e.g. `test_00074_tier1`) | Identifies which humanoid/image was shown. |
| `malice_tier` | int | `currentRound.tier` (`game_manifest.csv` `malice_tier`) | Independent variable. 4 confirmed levels: **`{0, 2, 8, 10}`** — `0` is the real-human control; `2`/`8`/`10` are the agent's malice budgets (Shane, PR #13's reward redesign; supersedes the earlier `{0, 1, 5, 10}`). See "Tier definition" below. |
| `round_index` | int | `station.roundIndex` | 0-based position of this trial in the shuffled deal order for the session (the shuffle is seeded off `content_session_key`, so order is reproducible per content set). Lets the analysis test order effects (fatigue/learning across the 30 rounds). |
| `condition` | enum: `human_alone` \| `human_ai` | chosen once on the condition-select screen (first visit only), persisted via `localStorage` | Assigned once per `participant_id` (between-subjects) — never changes across a participant's repeat sessions. Selected by the research team at session start (sessions are monitored via screenshare), not hashed client-side. See "Participant tracking" below. The Threat Advisory / AI Assistant panels are gated on this directly in `drawStationHUD()` (`human_ai` only) — this field records what was assigned and also fully determines whether the panels were shown. |
| `ground_truth_human` | `"human"` \| `"zombie"` | `currentRound.groundTruth` (`rounds.json` `groundTruth` / `game_manifest.csv` `ground_truth`, from `curate_game_sets.py`) | `"human"` if the subject is actually a healthy human (including injured-but-alive), `"zombie"` if overgrown/disguised. |
| `player_answer_human` | `"permit"` \| `"reject"` \| null | `permitRejectLabel()` applied to the argument passed to `recordAnswer()` | The player's literal checkpoint call, not a class judgment — `"permit"` = let the subject through, `"reject"` = turn them away. `null` = timed out, no decision made. |
| `correct` | bool \| null | `saysHuman === null ? null : saysHuman === (groundTruth === 'human')` | `null` only when the player timed out (`player_answer_human` is `null`). |
| `ai_confidence_shown` | float (0-100) | `currentRound.aiConfidence` (`game_manifest.csv` `ai_confidence_shown`, the CNN's `max_prob` — its probability for whichever class it predicted, not `cnn_p_healthy`) | The AI assistant's confidence — the real classifier's output, not an invented number. Post-PR #13 this trends *down* with tier (96/80/69% at tiers 2/8/10) rather than staying high, because the disguise now drives the CNN toward its decision boundary, so its winning-class probability falls toward 50%. |
| `cnn_says_human` | `"human"` \| `"zombie"` | `humanZombieLabel(currentRound.cnnSaysHuman)` (`rounds.json` `cnnSaysHuman`, from `curate_game_sets.py`) | The classifier's predicted class, labeled the same way as `ground_truth_human`/`player_answer_human`. Logged per trial so "did the player follow the AI" / over-trust is computable without joining `subject_id` back to the manifest — `ai_confidence_shown` alone is directionless. |
| `reaction_time_ms` | int | `performance.now() - roundStartTime` | Primary dependent variable alongside `correct`. |
| `zoom_level` | int (0-3) | `zoomLevel` state, set by the ZOOM 1 / ZOOM 2 / ZOOM 3 buttons | Which zoom setting was active when the player answered; `0` = no zoom. Replaces the old boolean `zoomed` field now that zoom has 3 discrete magnification levels instead of a single toggle. |
| `timestamp` | ISO 8601 string | `new Date().toISOString()` | Wall-clock time the trial was recorded. |

**Dropped fields (Claire):** `scanned`, `ai_assist_used`, `ai_assist_time_ms`,
and `flagged` are no longer logged or exported. All four had become
dead/fully-derived once the SCAN/AI ASSIST/FLAG buttons were removed (see the
update note above) — `condition` alone already tells you everything they
used to encode, so the redundant columns were cut from the trial-results CSV
export in `screen.html`. (The parallel `threeD_player/screen.html` prototype
has since been deleted from the repo — `screen.html` at the root is the only
game file.)

## Data quality field (added by analysis, not the UI)

| Field | Type | Description |
|---|---|---|
| `flagged_invalid` | bool | Set by the analysis pipeline (see `analysis/data_quality.py`), not the game itself — `reaction_time_ms < MIN_VALID_RESPONSE_MS` (default 1000ms) or `player_answer_human is None`. Kept separate from the player-facing `flagged` field, which means something different (player suspicion, not data validity). |

## Tier definition — resolved again (Shane, PR #13; supersedes PR #6/#7)

**Current confirmed tier set: `{0, 2, 8, 10}`.** `tier 0` is the real-human
control; `2`/`8`/`10` are the trained agent's **malice budgets** (not edit-count
snapshots — the previous scheme). The tiers are produced by running the trained
greedy policy at budget 2 / 8 / 10 and letting it **stop on its own**. Control
photos are tilted by the same small angles as the zombies so "is it tilted?"
can't leak the answer. Verified directly against
`data/game_sets/game_manifest.csv` (180 rows, `malice_tier` values exactly
`{0, 2, 8, 10}`). This supersedes the `{0, 1, 5, 10}` set from PR #7 (and
`{0, 1, 3, 6}` before that). Full rationale in Shane's
`docs/rl-red-team-design.md`.

**What changed in PR #13 — the AI now grows *uncertain*, not confidently
wrong.** The old reward maximized P(healthy), which pushed the disguise past the
classifier's decision boundary so the top tier *fully fooled* the CNN
(confidently, wrongly "healthy" ~88% of the time). The new reward is a "tent"
peaking at the decision boundary (P(healthy) ≈ 0.45), and the agent is trained
to **stop at the edge of the verdict**. So at the top tier the figure looks
human to the eye but the AI is visibly unsure (leans "zombie" at only ~55%),
rather than confidently agreeing it's human. This is the whole point: a
confident-but-wrong AI gives the player nothing to decide (eyes and AI agree),
whereas a *visibly uncertain* AI is a genuine dilemma.

Reliability across tiers (from Shane's `docs/rl-red-team-design.md`; mean AI
confidence independently re-derived from `game_manifest.csv`'s
`ai_confidence_shown` and matches — 96/80/69):

| Tier (budget) | mean AI confidence | AI uncertain | confidently wrong |
|---|---|---|---|
| 2 (low) | 96% | 3% | 10% (CNN baseline error) |
| 8 (mid) | 80% | 40% | 0% |
| 10 (high) | 69% | 63% | 7% |

The old design's ~88% confidently-wrong rate at the top tier is gone;
confidently-wrong now stays at the CNN's ~7-10% baseline across all tiers.
That reframes the `human_alone` vs `human_ai` comparison: the question is no
longer "does the human blindly follow an AI that's confidently wrong?" but
"when the AI signals its own uncertainty at high tiers, does that *help* the
human decide, or do they just disregard a low-confidence reading?" Note the
per-tier `cnn_correct` accuracy is noisy and can look non-monotonic (mid tier
can beat low) because each zombie appears at only one tier — a between-source
comparison, per the limitation Shane documents — so `ai_confidence`/`uncertain`
are the meaningful reliability signals, not raw per-tier accuracy.

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
  both conditions would contaminate the comparison). **Assignment is now
  manual**, superseding the earlier hash-the-UUID-mod-2 scheme: the first page
  load shows a full-screen condition-select (above the "Click to start"
  blocker), and since every session is run monitored over screenshare, the
  research team picks the mode for the participant. That puts the 50/50 split
  under the team's direct control (keep a running tally per condition while
  recruiting) instead of leaving it to hash luck — but it also means balance
  is now a *protocol* obligation, not a code guarantee: still check and report
  the realized split after data collection. A returning `participant_id` skips
  the selector entirely (`condition` is already in `localStorage`).
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

1. ~~`data/game_sets/rounds.json` wiring / SCAN/AI ASSIST respecting
   `condition`~~ — `screen.html` now fetches `rounds.json` directly (see the
   `fetch('data/game_sets/rounds.json')` call), and the "disabled entirely in
   `human_alone`" requirement is resolved by removing the SCAN/AI ASSIST/FLAG
   buttons altogether: the Threat Advisory and AI Assistant panels render
   automatically for `human_ai` and never for `human_alone`, so there's no
   button state left to gate. See the schema update note at the top of this
   doc and the `scanned`/`ai_assist_used`/`zoom_level`/`flagged` field rows
   above for the resulting field-level changes.
2. ~~`game_set` not yet exported~~ — resolved: it ships as the
   `content_session_key` column (see the field table above), assigned by
   `getNextContentSession()` and included in both the CSV export and the
   Sheet POST.
3. ~~`tests/test_scorekeeper.py::test_save_log_log_csv_columns`~~ — resolved;
   the full suite passes.

## Survey data (pre/post AI-attitude surveys)

Added in the survey PR (`screen.html`). Separate from the trial CSV — this is
participant-level, not trial-level — but joins to it on `participant_id` +
`session_id`. Only the **AI-assist** condition produces survey data; the
`human_alone` condition takes no survey (there is no AI to form attitudes
about). Both surveys are 7-point Likert (1 = strongly disagree … 7 = strongly
agree) and are shown **once per participant**: the pre survey before the first
game, the post survey after it. `exportSurveyCSV()` downloads
`survey-<participant_id>-<timestamp>.csv` in **long format**, one row per
answered question:

| column | meaning |
| --- | --- |
| `participant_id` | joins to the trial CSV |
| `session_id` | the play session the survey bracketed |
| `condition` | always `human_ai` (the only condition surveyed) |
| `phase` | `pre` or `post` |
| `question_id` | stable id: `pre_q1`..`pre_q10`, `post_q1`..`post_q10` |
| `question_text` | full item wording (CSV-escaped; several contain commas) |
| `response` | integer 1–7 |
| `timestamp` | ISO-8601 — `pre` rows carry the time the pre survey was actually submitted (stamped at submit, persisted in `localStorage`); `post` rows carry the post-survey submit/export time |

The pre items measure general AI trust/reliance attitudes; the post items
measure the in-game AI experience. Both are required (a participant cannot
advance past a survey until every item is answered), so there are no partial
rows. Analysis idea: pair `pre`/`post` per participant to measure attitude
shift, and correlate post reliance items against that participant's trial-level
accuracy and AI-agreement rate (`player_answer_human` mapped `permit`->`human`/
`reject`->`zombie` and compared against `cnn_says_human` — the values no
longer match as literal strings since the 2026-07-25 `permit`/`reject`
change above; the `ai_assist_used` field this originally referenced was
dropped).
