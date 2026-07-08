from gameplay.enums import State

# Convenience mappings to make converting these throughout the code less error prone
MAP_CLASS_STR_TO_INT: dict[str, int] = {s.value: i for i, s in enumerate(State)}
MAP_CLASS_INT_TO_STR: tuple[str, ...] = tuple(s.value for s in State)


class Humanoid:
    """A single humanoid entity encountered during the game.

    Each humanoid has a ground-truth state (zombie, healthy, injured, or corpse)
    and an associated image file path. The player must classify and act on each
    humanoid before time runs out.

    Attributes:
        fp: Relative file path to the humanoid's image.
        state: Ground-truth state string (one of the :class:`~gameplay.enums.State`
            enum values).
        value: Numeric priority score in [0.0, 1.0]. Assigned by DataParser; higher = higher triage priority.
    """

    fp: str
    state: str
    value: float

    def __init__(self, fp: str, state: str, value: float = 0.0) -> None:
        """Initialize a Humanoid.

        Args:
            fp: Relative file path to the humanoid's image.
            state: Ground-truth state string (e.g. ``"zombie"``, ``"healthy"``).
            value: Numeric priority score in [0.0, 1.0]. Assigned randomly by DataParser; higher = higher triage priority.
        """
        self.fp = fp
        self.state = state
        self.value = value

    def is_zombie(self) -> bool:
        """Return True if this humanoid is a zombie."""
        return self.state == State.ZOMBIE.value

    def is_injured(self) -> bool:
        """Return True if this humanoid is injured."""
        return self.state == State.INJURED.value

    def is_healthy(self) -> bool:
        """Return True if this humanoid is healthy."""
        return self.state == State.HEALTHY.value

    def is_corpse(self) -> bool:
        """Return True if this humanoid is a corpse."""
        return self.state == State.CORPSE.value

    @staticmethod
    def get_state_idx(class_string: str) -> int:
        """Return the integer index for a state string.

        Args:
            class_string: A :class:`~gameplay.enums.State` enum value string
                (e.g. ``"zombie"``).

        Returns:
            Integer index corresponding to the state in enum order.
        """
        return MAP_CLASS_STR_TO_INT[class_string]

    @staticmethod
    def get_state_string(class_idx: int) -> str:
        """Return the state string for an integer index.

        Args:
            class_idx: Integer index of the state in enum order.

        Returns:
            State string (e.g. ``"zombie"``).
        """
        return MAP_CLASS_INT_TO_STR[class_idx]

    @staticmethod
    def get_all_states() -> list[str]:
        """Return all state strings defined in the :class:`~gameplay.enums.State` enum.

        Returns:
            A copy of the state string list in enum declaration order.
            Returns a copy so callers cannot mutate the module-level mapping.
        """
        return list(MAP_CLASS_INT_TO_STR)
