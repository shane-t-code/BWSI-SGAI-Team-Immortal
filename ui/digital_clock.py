"""Digital countdown clock — shows remaining game time in minutes.

This is a placeholder implementation. The public interface matches ui/clock.py
exactly so that LEGACY_CLOCK = False works end-to-end. The visual display is
intentionally minimal — replace it with a proper Tkinter widget.

Interface contract (must match ui/clock.py):
    __init__(root, w, h, init_h, init_m)
    update_time(h, m)

The h, m arguments use the same elapsed-time encoding as the analog clock:
    h = 12 - floor(remaining_time / 60)
    m = (60 - (remaining_time % 60)) % 60
Remaining minutes = (12 - h) * 60 - m
"""

import math
import tkinter as tk

from gameplay.constants import UI_COLORS


class DigitalClock:
    """Minimal digital countdown clock that shows remaining minutes as text.

    Implements the same public interface as :class:`~ui.clock.Clock` so it
    can be swapped in transparently via the ``LEGACY_CLOCK`` feature flag.

    Attributes:
        canvas: Tkinter canvas containing the countdown label.
        _label: Label widget displaying the remaining-time string.
    """

    canvas: tk.Canvas
    _label: tk.Label

    def __init__(self, root: tk.Misc, w: int, h: int, init_h: int, init_m: int) -> None:
        """Create the digital clock canvas and display the initial time.

        Args:
            root: Tkinter parent widget.
            w: Window width in pixels (used to position and size the canvas).
            h: Window height in pixels (used to size the canvas).
            init_h: Initial hour value (same encoding as :class:`~ui.clock.Clock`).
            init_m: Initial minute value (same encoding as :class:`~ui.clock.Clock`).
        """
        self.canvas = tk.Canvas(
            root,
            width=math.floor(0.2 * w),
            height=math.floor(0.3 * h),
            bg=UI_COLORS["panel"],
            highlightthickness=0,
        )
        self.canvas.place(x=math.floor(0.75 * w), y=50)

        self._label = tk.Label(
            self.canvas,
            text="",
            font=("Arial", 24, "bold"),
            justify=tk.CENTER,
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        )
        self._label.place(x=0, y=80, width=math.floor(0.2 * w))

        self.update_time(init_h, init_m)

    def update_time(self, h: int, m: int) -> None:
        """Update the countdown label to show remaining minutes.

        Args:
            h: Elapsed hours from midnight (same encoding as Clock).
            m: Elapsed minutes within the current hour (same encoding as Clock).
        """
        remaining = max(0, (12 - h) * 60 - m)  # clamp: scram overshoot can go negative
        self._label.config(text=f"{remaining} min\nremaining")
