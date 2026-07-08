from enum import Enum


class State(Enum):
    """Ground-truth humanoid states.

    These values correspond to the ``state`` column in the metadata CSV and to
    the ``state`` attribute on :class:`~gameplay.humanoid.Humanoid`.
    """

    ZOMBIE = "zombie"
    HEALTHY = "healthy"
    INJURED = "injured"
    CORPSE = "corpse"


class ActionState(Enum):
    """Player actions available each turn."""

    SAVE = "save"
    SQUISH = "squish"
    SKIP = "skip"
    SCRAM = "scram"


# COUPLING: ActionState and ActionCost must stay in sync — same members, same order.
# Adding an action to one without the other produces a silent gap in cost lookups.
class ActionCost(Enum):
    """Time cost in minutes for each player action.

    Design intent: SKIP (15 min) costs more than SQUISH (5 min) because skip
    represents navigating around the humanoid while leaving them behind — more
    time-consuming than a quick squish. This asymmetry matters for reward shaping:
    squishing is faster even when it kills, so the RL agent must learn to balance
    speed against humanitarian outcomes rather than defaulting to squish.
    """

    SAVE = 30
    SQUISH = 5
    SKIP = 15
    SCRAM = 120
