import tkinter as tk
from collections.abc import Callable

from gameplay.constants import UI_COLORS

# Map base action name → button background color.
_BTN_COLORS: dict[str, str] = {
    "Skip": UI_COLORS["btn_skip"],
    "Squish": UI_COLORS["btn_squish"],
    "Save": UI_COLORS["btn_save"],
    "Scram": UI_COLORS["btn_scram"],
}


class ButtonMenu:
    """Renders the action button row and manages button enabled/disabled state.

    In human modes, four buttons are displayed (Skip, Squish, Save, Scram),
    keyed by base action name. In machine mode, a single "Advance turn" button
    is displayed.

    Attributes:
        canvas: Tkinter canvas containing the action buttons.
        buttons: Dict mapping base action name (e.g. ``"Save"``) to its
            :class:`tk.Button`. In machine mode the single button is keyed
            by ``"Advance turn"``.
    """

    canvas: tk.Canvas
    buttons: dict[str, tk.Button]

    def __init__(
        self, root: tk.Misc, items: list[tuple[str, Callable[[], None]]]
    ) -> None:
        """Create the button canvas and render all buttons.

        Args:
            root: Tkinter parent widget.
            items: List of ``(display_text, callback)`` pairs. Display text
                may include a cost suffix (e.g. ``"Save [-30 min]"``); the
                base name is extracted for dict keying.
        """
        # height=80 is sized for up to 4 buttons (each ~20px). Adding more buttons
        # without increasing this value causes them to overflow below the canvas
        # boundary and overlap with widgets placed below y=150+80=230.
        self.canvas = tk.Canvas(
            root,
            width=500,
            height=80,
            bg=UI_COLORS["bg"],
            highlightthickness=0,
        )
        self.canvas.place(x=100, y=150)
        self.buttons = self._create_buttons(self.canvas, items)
        self._create_menu(self.canvas, self.buttons)

    def disable_all(self) -> None:
        """Unconditionally disable all buttons.

        Call at the top of every action callback to prevent double-click
        re-entrance. ``disable_buttons`` will re-enable valid buttons at
        the end of the turn.
        """
        for btn in self.buttons.values():
            btn.config(state="disabled")

    def disable_buttons(
        self,
        remaining_time: int,
        remaining_humanoids: int,
        at_capacity: bool,
    ) -> None:
        """Enable or disable buttons based on current game state.

        All buttons are disabled at game-over (time or humanoids exhausted).
        During active play, Save is disabled when the ambulance is full;
        all other buttons remain available.

        Args:
            remaining_time: Minutes left in the shift.
            remaining_humanoids: Number of unvisited humanoids remaining.
            at_capacity: Whether the ambulance is currently full.
        """
        # Game over: disable all buttons.
        if remaining_humanoids == 0 or remaining_time <= 0:
            for btn in self.buttons.values():
                btn.config(state="disabled")
            return

        # Re-enable all buttons first, then apply state-based constraints.
        for btn in self.buttons.values():
            btn.config(state="normal")

        # SAVE is disabled when at capacity. All others remain available.
        # "Save" key is absent in machine mode — guard skips capacity logic cleanly.
        if at_capacity and "Save" in self.buttons:
            self.buttons["Save"].config(state="disabled")

    @staticmethod
    def _create_buttons(
        canvas: tk.Canvas,
        items: list[tuple[str, Callable[[], None]]],
    ) -> dict[str, tk.Button]:
        """Create and return a dict of styled Tkinter buttons.

        Args:
            canvas: Parent canvas to attach the buttons to.
            items: List of ``(display_text, callback)`` pairs. Cost suffixes
                (e.g. ``" [-30 min]"``) are stripped when building the dict key.

        Returns:
            Dict mapping base action name to :class:`tk.Button`.
        """
        buttons: dict[str, tk.Button] = {}
        for text, action in items:
            base = text.split(" [")[0]  # strip cost suffix to get action name
            bg = _BTN_COLORS.get(base, UI_COLORS["btn_default"])
            buttons[base] = tk.Button(
                canvas,
                text=text,
                height=2,
                width=15,
                command=action,
                bg=bg,
                fg=UI_COLORS["fg"],
                activebackground=UI_COLORS["btn_active"],
                activeforeground=UI_COLORS["fg"],
                disabledforeground="#555555",
                relief=tk.FLAT,
                bd=1,
            )
        return buttons

    @staticmethod
    def _create_menu(canvas: tk.Canvas, buttons: dict[str, tk.Button]) -> None:
        """Pack all buttons into the canvas.

        Args:
            canvas: Parent canvas to attach the buttons to.
            buttons: Dict of buttons to pack, in the order they should appear.
        """
        for button in buttons.values():
            button.pack(side=tk.TOP, pady=10)
