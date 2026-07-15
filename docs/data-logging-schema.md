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
| `participant_id` | string | assigned at session start | Not yet in `recordAnswer()` — needs to be added once real sessions run (see Open Questions). |
| `session_id` | string | assigned at session start | Groups all trials from one sitting; lets us tell repeat participants apart from repeat sessions. |
| `subject_id` | string | `currentRound.subjectId` | Identifies which humanoid/image was shown. |
| `malice_tier` | int | `currentRound.tier` | Independent variable. See "Tier definition" below — **currently mismatched between Shane's and Claire's code, needs reconciling.** |
| `condition` | enum: `human_alone` \| `human_ai` | not yet in `recordAnswer()` | Needs to be an assigned condition, not inferred from whether the player happened to click AI ASSIST. See Open Questions. |
| `ground_truth_human` | bool | `currentRound.groundTruthIsHuman` | True if the subject is actually a healthy human, false if overgrown/zombie. |
| `player_answer_human` | bool \| null | argument to `recordAnswer()` | `null` = timed out, no decision made. |
| `correct` | bool \| null | computed: `player_answer_human === ground_truth_human` | `null` when `player_answer_human` is `null`. |
| `ai_confidence_shown` | float (0-100) | `currentRound.aiConfidence` | The AI assistant's confidence, if/when shown. |
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

## Tier definition — needs reconciling (flag for Shane + Claire)

Shane's `AdversarialEnv` (`game_interfaces/adversarial_env.py`) defines tiers as
**edit-count snapshots along a single budget-6 rollout** — the comment at
`MAX_BUDGET = 6` says tiers are made "by snapshotting the image after 1 / 3 / 6
edits of a single budget-6 rollout," implying 3 tiers: `{1, 3, 6}`.

Claire's placeholder `ROUNDS` array in `index.html` uses `tier` values `{0, 1,
3, 4}`, where `0` appears to mean "no disguise, real human." That `4` doesn't
match Shane's `{1, 3, 6}` scheme, and it's explicitly placeholder data ("replace
with your real per-tier manifest" per the comment above `ROUNDS`), so this may
just not have been updated yet — but it needs to be before real trials run.

**Recommendation to bring to the team:** four tiers total — `tier 0` = real
human (no disguise, control condition) plus Shane's three malice tiers `{1, 3,
6}` — rather than inventing a fourth malice tier. I'll raise this with Shane
and Claire directly rather than silently picking one in code, since it affects
both the RL snapshot script and the game manifest.

## Open questions (blocking, need a team decision before real trials run)

1. **Condition assignment.** Right now AI ASSIST is a per-round button the
   player can click or not — that's an observational variable, not an
   assigned condition. For a clean `human_alone` vs. `human_ai` comparison, the
   condition should be **fixed for a whole session (or block of rounds)**,
   with the AI panel either never available or always auto-revealed,
   independent of whether the player would have clicked it. This is a game-mode
   design decision, not something I should silently patch into someone else's
   UI file — raising it for Claire/the team to weigh in on.
2. **`participant_id` / `session_id`.** Not generated anywhere yet. Needs a
   simple prompt or config at game launch before any real data collection.
3. **Where trial data is actually persisted.** As of this branch, `results` in
   `index.html` only ever gets `console.log`'d — it's never saved to a file.
   I added a CSV export (see `docs/data-logging-schema.md` companion change in
   `index.html` — `exportResultsCSV()`) so a session's data can actually be
   retrieved. This is additive only; it does not touch `PERMIT`/`REJECT`
   wiring, which is intentionally still stubbed out pending real images.
