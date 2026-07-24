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
than gating individual buttons was. `scanned` / `ai_assist_used` /
`ai_assist_time_ms` are kept in the schema but are now derived from
`condition` rather than recorded from a click; `flagged` has no button behind
it anymore and is always `false`. The old boolean `zoomed` field was replaced
by `zoom_level` (int 0-3) since zoom is now 3 discrete magnification settings
(ZOOM 1/2/3) instead of a single on/off toggle. See the field table below for
the updated per-field detail.

## One row per trial (one humanoid judged by one participant)

| Field | Type | Source | Description |
|---|---|---|---|
| `participant_id` | string | `getOrCreateParticipantId()` in `index.html`, persisted via `localStorage` | Identifies one person across repeat plays on the same browser. See "Participant tracking" below. |
| `session_id` | string | generated fresh each page load (`crypto.randomUUID()`) | One sitting/playthrough — a participant who replays gets a new `session_id` each time but the same `participant_id`. |
| `game_set` | int (1-6) | `game_manifest.csv` / `rounds.json` `session` column | Which of Shane's 6 disjoint 30-round content sets was shown. **Not the same thing as `session_id` above** — `game_set` identifies the content shown, `session_id` identifies the sitting. Naming collision with the raw CSV's `session` column called out explicitly so it doesn't get confused with a participant's sitting. |
| `subject_id` | string | `currentRound.subjectId` (`game_manifest.csv` `subject_id`, e.g. `test_00074_tier1`) | Identifies which humanoid/image was shown. |
| `malice_tier` | int | `currentRound.tier` (`game_manifest.csv` `malice_tier`) | Independent variable. 4 confirmed levels: **`{0, 2, 8, 10}`** — `0` is the real-human control; `2`/`8`/`10` are the agent's malice budgets (Shane, PR #13's reward redesign; supersedes the earlier `{0, 1, 5, 10}`). See "Tier definition" below. |
| `condition` | enum: `human_alone` \| `human_ai` | `getOrCreateParticipantId()` bootstrap in `index.html`, via `analysis/trial_protocol.assign_condition()` logic mirrored client-side | Assigned once per `participant_id` (between-subjects) — never changes across a participant's repeat sessions. See "Participant tracking" below. The Threat Advisory / AI Assistant panels are now gated on this directly in `drawStationHUD()` (`human_ai` only), rather than via disabled SCAN/AI ASSIST buttons — this field records what was assigned and, since the removal of those buttons, also fully determines whether the panels were shown. |
| `ground_truth_human` | bool | `currentRound.groundTruthIsHuman` | True if the subject is actually a healthy human, false if overgrown/zombie. |
| `player_answer_human` | bool \| null | argument to `recordAnswer()` | `null` = timed out, no decision made. |
| `correct` | bool \| null | computed: `player_answer_human === ground_truth_human` | `null` when `player_answer_human` is `null`. |
| `ai_confidence_shown` | float (0-100) | `currentRound.aiConfidence` (`game_manifest.csv` `ai_confidence_shown`, the CNN's `max_prob` — its probability for whichever class it predicted, not `cnn_p_healthy`) | The AI assistant's confidence — the real classifier's output, not an invented number. Post-PR #13 this trends *down* with tier (96/80/69% at tiers 2/8/10) rather than staying high, because the disguise now drives the CNN toward its decision boundary, so its winning-class probability falls toward 50%. |
| `reaction_time_ms` | int | `performance.now() - roundStartTime` | Primary dependent variable alongside `correct`. |
| `scanned` | bool | `condition === 'human_ai'` (SCAN button removed; no longer a player toggle) | Whether the Threat Advisory panel was shown this round — `true` for every `human_ai` trial, `false` for every `human_alone` trial. |
| `zoom_level` | int (0-3) | `zoomLevel` state, set by the ZOOM 1 / ZOOM 2 / ZOOM 3 buttons | Which zoom setting was active when the player answered; `0` = no zoom. Replaces the old boolean `zoomed` field now that zoom has 3 discrete magnification levels instead of a single toggle. |
| `ai_assist_used` | bool | `condition === 'human_ai'` (AI ASSIST button removed; no longer a player toggle) | Whether the AI Assistant/confidence panel was shown this round — same automatic-display logic as `scanned` above. |
| `ai_assist_time_ms` | int \| null | `condition === 'human_ai' ? 0 : null` | `0` for `human_ai` trials, since the panel is visible from the moment the round starts (no click to time anymore); `null` for `human_alone`. No longer measures a meaningful "time to open" — kept for schema/column-count stability, but analysis should not treat it as a real latency. |
| `flagged` | bool | Always `false` — FLAG button removed | Dead field, kept only so the CSV column order/count doesn't change. Player marking-for-review is no longer collected. |
| `timestamp` | ISO 8601 string | `new Date().toISOString()` | Wall-clock time the trial was recorded. |

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
  both conditions would contaminate the comparison). Assigned client-side by
  hashing the `participant_id` UUID and taking it mod 2.
  **Balance caveat (Shane, PR #7 review):** this is *not* equivalent to
  `analysis/trial_protocol.assign_condition()`, despite the earlier claim that
  it "mirrors" it. The Python version alternates on an enrollment index, which
  *guarantees* a near-50/50 split. The browser version has no shared enrollment
  counter (static page, independent browsers, no backend), so it hashes the
  UUID instead — which is only 50/50 *in expectation*. With ~30-40 participants
  a lopsided split (e.g. 24/13) can happen by chance. **Mitigation:** check the
  realized `condition` split after data collection and report it; if it's badly
  skewed, weight or note it in the analysis. A guaranteed balance would require
  assigning `condition` at signup from a shared counter (backend or manual
  enrollment sheet) rather than client-side hashing — out of scope for the MVP,
  but the fallback if the observed split is unacceptable.
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
2. `game_set` is defined in the schema above but **not yet exported** by
   `exportResultsCSV()` in `index.html`. The blocker is no longer the data
   (`rounds.json` carries the `session`/game-set column now) — it's just Claire's
   wiring: the moment `ROUNDS` is fed from `rounds.json`, add a `game_set` column
   to the `exportResultsCSV()` header + row arrays so it reaches the trial CSV.
3. Prajwal still needs to fix `tests/test_scorekeeper.py::test_save_log_log_csv_columns`
   (missing `confidence_score` in `LOG_COLUMNS`) per Shane's note — unrelated
   to this pipeline, flagging here only because it's a known pre-existing
   failure noted twice now.

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
| `timestamp` | ISO-8601, when the post survey was submitted/exported |

The pre items measure general AI trust/reliance attitudes; the post items
measure the in-game AI experience. Both are required (a participant cannot
advance past a survey until every item is answered), so there are no partial
rows. Analysis idea: pair `pre`/`post` per participant to measure attitude
shift, and correlate post reliance items against that participant's trial-level
`ai_assist_used` rate and accuracy.
