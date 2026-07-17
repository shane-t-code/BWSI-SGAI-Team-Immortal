# Trial Protocol — Malice-Budget Experiment

Owner: Smit (data lead). Companion to `docs/data-logging-schema.md`.

## Design

- **Independent variable:** `malice_tier` — 4 levels: `0` (real human, control),
  `1`, `5`, `10` (malice-budget edit counts — updated by Shane in PR #7 after
  retraining the two-channel recolor+rotation agent; supersedes the earlier
  `{0, 1, 3, 6}` from PR #6).
- **Between-subjects factor:** `condition` — `human_alone` vs. `human_ai`,
  **resolved as between-subjects per participant** (not per session): a
  participant is assigned one condition the first time they're seen and keeps
  it across every repeat session, per Shane's note that AI advice teaches
  people what a disguise looks like — letting someone see both conditions
  would let that leak into the "alone" condition on a later session. See
  "Participant tracking" in `docs/data-logging-schema.md` for the assignment
  mechanism (`participant_id` created + condition assigned once,
  `localStorage`-persisted).
- **Dependent variables:** `correct` (accuracy) and `reaction_time_ms`
  (response time), both broken out by tier and by condition.
- **Primary vs. exploratory analysis:** the primary comparison uses only each
  participant's *first* session (`analysis.analyze_results.filter_first_session()`)
  — pooling every repeat session at the trial level would overweight whoever
  replayed the most. Repeat sessions remain in the raw data as exploratory
  material (e.g. "does practice beat the disguise?").

## Sample size

Target for the MVP playtest (per Shane's PR #7 numbers): **15-20 participants
per condition**, each doing **5 trials per malice tier** (4 tiers). That's
`15 x 5 = 75` to `20 x 5 = 100` trials per (tier, condition) cell —
comfortably clears the `>30`-trials-per-cell rule-of-thumb floor for a
proportion (accuracy) comparison, about 2-3x over. Not a formal power
calculation (no pilot variance estimate exists yet) — revisit once pilot data
exists and an effect size can be estimated directly.

## Randomization

Within a session, tier order (and which specific subject image appears at
each tier) must be randomized per participant so practice effects don't
systematically favor later tiers. `analysis/trial_protocol.py` implements
this: `build_trial_order()` takes a list of tiers, a repeats-per-tier count,
and a seed, and returns a shuffled trial order (one row per trial, in
presentation order) — deterministic per seed so a session can be replayed for
debugging.

## Data quality

See `analysis/data_quality.py`. A trial is flagged invalid if:
- `reaction_time_ms < MIN_VALID_RESPONSE_MS` (default 1000ms) — too fast to
  have actually looked at the image.
- `player_answer_human is None` — timed out, no real judgment made.

Flagged trials are kept in the raw log (never deleted) but excluded from the
primary accuracy/response-time analysis, with a reported count of how many
were dropped per condition so we can sanity-check the flag rate isn't
suspiciously high.
