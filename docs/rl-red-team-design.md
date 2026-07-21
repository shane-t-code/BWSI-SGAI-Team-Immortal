# Red Team RL — Design Doc

The Red Team is the reinforcement-learning agent that disguises zombie images for the
study. This is the reference explanation: what it does and why, in plain terms first,
then in the proper vocabulary.

## In plain terms

The Red Team is a small AI that disguises zombie photos so they're hard to judge. It
works in two steps on each zombie. **First** it repaints the zombie's bright-green skin
to look like human skin — this is what fools a *person* looking at the picture.
**Second** it rotates the whole photo a few degrees — a human barely notices (it just
looks like a tilted camera), but it confuses the zombie-detecting computer-vision model
(the "AI assistant" the player can consult), because that model was only ever trained on
perfectly upright photos. The disguise gets stronger the more "malice budget" the Red
Team is allowed: at low budget it only lightly recolors (still obviously a zombie); at
medium budget it fully recolors (looks human) but the AI is still fairly sure it's a
zombie; and at the top budget it fully recolors *and* tilts just enough that the AI
becomes genuinely unsure — it still leans "zombie" but only about 55% confident. The
important part: the Red Team is trained to **stop** tilting right at that unsure point,
so it never over-tilts to where the AI flips to confidently (and wrongly) saying "healthy
human." That confident-wrong outcome is useless for the study, because then the player's
eyes and the AI would agree it's human and there'd be no real decision to make. The three
difficulty tiers players see are just this same agent run at three budgets (2, 8, 10).
Real human photos are mixed in as controls, tilted by the same small amounts as the
zombies, so "is it tilted?" never gives away the answer.

## In RL terms

Technically this is a **Deep Q-Network (DQN)** agent in a custom **Gymnasium
environment**. The **action space** is four discrete moves — rotate CW, rotate CCW,
recolor, and stop — and the **observation** is a 9-dimensional vector: cumulative angle,
cumulative recolor, remaining budget, the CNN's four class probabilities, and two
rotation **lookahead** features (what P(healthy) would be one step each way). The
**reward** is a **potential-based delta reward** that telescopes to a weighted "disguise
score" with two channels. The *human* channel maximizes `hue_proxy` (how skin-toned the
figure looks). The *AI* channel is a **tent function** that peaks when the CNN's
P(healthy) reaches a target of **0.45 — the classifier's decision boundary** — and
decreases on both sides, so overshooting past the boundary is penalized and the agent
**learns to stop at the edge of the verdict**. Constraints are enforced by **action
masking**, not reward penalties: rotation is masked until recolor is complete (the
**two-phase disguise**), and the fixed **budget of 10 is a guardrail** that physically
prevents the agent from over-rotating into confidently-wrong territory. The **malice
budget is the study's independent variable**; the tiers are produced by running the
trained **greedy policy** at budgets 2 / 8 / 10 and letting it **STOP** on its own.
Relevant concepts to name: **off-the-data-manifold** (why a small tilt breaks a CNN that
only saw upright images), **orthogonal attack channels** (color fools the human, geometry
fools the AI — independent axes), **decision-boundary targeting**, and **between-source
tier assignment** (each underlying render appears at only one tier/session so repeat
players never re-see a scene).

## Where it lands (curated set)

| tier | budget | mean AI confidence | AI uncertain | confidently wrong |
|------|--------|--------------------|--------------|-------------------|
| low  | 2      | 96%                | 3%           | 10% (CNN baseline error) |
| mid  | 8      | 80%                | 40%          | 0% |
| high | 10     | 69%                | 63%          | 7% |

The 88% confidently-wrong rate of the old design (which fully fooled the AI at the top
tier) is gone. Malice now converts AI *confidence* into *doubt*, not into confident
errors — the confidently-wrong rate stays at the CNN's ~7-10% baseline across all tiers.

## Designs that were tried and rejected

- **Maximize P(healthy)** (original): drove the agent past the boundary → 88% confidently
  wrong at the top tier. Replaced by the boundary-targeting tent.
- **Maximize raw uncertainty (1 - max_prob)**: too flat a learning signal far from the
  torn point, so the agent under-rotated → fewer uncertain images (40% vs 57%).
- **Target P(zombie)=0.5 with a wider budget (16)**: the extra rotation room let the
  agent overshoot the boundary → dead cells doubled to 13%. The budget-10 cap is the fix.

## Known limitations (accepted)

- **Pale figures** keep a faint green tint on the feet at all tiers (their skin is
  colorimetrically identical to olive clothing, so no per-pixel color rule separates
  them). Accepted for now; can be handled at curation later by assigning them to low
  tiers or removing them.
- **Between-source tiers**: because each zombie appears at only one tier (repeat-player
  safety), per-tier reliability compares *different* zombies, so small-sample noise can
  make the mid tier look easier than the low tier even though the full-pool trend is
  monotonic.
