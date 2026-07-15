# Trial Protocol — Malice-Budget Experiment

Owner: Smit (data lead). Companion to `docs/data-logging-schema.md`.

## Design

- **Independent variable:** `malice_tier` — 4 levels: `0` (real human, control),
  `1`, `3`, `6` (malice-budget edit counts, pending confirmation from Shane —
  see the tier mismatch flagged in `data-logging-schema.md`).
- **Between/within factor:** `condition` — `human_alone` vs. `human_ai`.
  Recommended as **between-subjects per session** (a participant is either an
  "alone" participant or an "assisted" participant for their whole session) so
  the AI panel's presence doesn't contaminate a within-session comparison —
  learning to rely on / distrust the AI mid-session would confound a
  within-subjects design. Open for team discussion if a within-subjects
  design is preferred instead (each participant does both, in counterbalanced
  blocks) — that would need `condition` recorded per-block rather than per
  session and adds AB/BA order counterbalancing on top of tier randomization.
- **Dependent variables:** `correct` (accuracy) and `reaction_time_ms`
  (response time), both broken out by tier and by condition.

## Sample size

Not yet computed against a power analysis (no pilot variance estimate exists
yet). As a starting point for the MVP: **~15-20 trials per tier per
condition per participant**, with **enough participants that each
tier x condition cell has at least ~30 trials total** once pooled across
participants — a common rule-of-thumb floor for a proportion (accuracy)
comparison, not a formal power calculation. Revisit once pilot data exists
and we can estimate the effect size we're actually seeing.

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
