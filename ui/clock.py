import math
import os
from typing import Any
import tkinter as tk

from gameplay.constants import UI_COLORS


class Clock:
    """Analog clock widget that displays elapsed game time.

    The clock counts up from midnight, so the hour hand shows elapsed hours
    and the minute hand shows elapsed minutes within the current hour.

    Attributes:
        canvas: Tkinter canvas containing the clock image and hands.
        sticks: Canvas item IDs for the two clock hands (hour, minute),
            or ``None`` before :meth:`generate_clock_hands` is called.
        image: Background clock-face image, or ``None`` before
            :meth:`generate_bg` is called.
        x: X-coordinate of the clock center in canvas pixels.
        y: Y-coordinate of the clock center in canvas pixels.
        length: Hand lengths in pixels: ``[hour_length, minute_length]``.
        width: Hand widths in pixels: ``[hour_width, minute_width]``.
        fill: Hand colors: ``[hour_color, minute_color]``.
    """

    canvas: tk.Canvas
    sticks: list[int] | None
    image: Any | None  # tk.PhotoImage
    x: int
    y: int
    length: list[int]
    width: list[int]
    fill: list[str]

    def __init__(self, root: Any, w: int, h: int, init_h: int, init_m: int) -> None:
        """Create the clock canvas and render the initial time.

        Args:
            root: Tkinter parent widget.
            w: Window width in pixels (used to position the canvas).
            h: Window height in pixels (used to size the canvas).
            init_h: Initial hour value (elapsed hours from midnight).
            init_m: Initial minute value (elapsed minutes within the hour).
        """
        self.canvas = tk.Canvas(
            root,
            width=math.floor(0.2 * w),
            height=math.floor(0.3 * h),
            bg=UI_COLORS["panel"],
            highlightthickness=0,
        )
        self.canvas.place(x=math.floor(0.75 * w), y=50)
        self.sticks = None
        self.image = None
        canvas_w = math.floor(0.2 * w)
        canvas_h = math.floor(0.3 * h)
        self.x = canvas_w // 2  # Center Point x
        self.y = canvas_h // 2  # Center Point y
        self.length = [30, 55]  # Stick Length
        self.width = [3, 6]
        self.fill = [UI_COLORS["accent"], "#777777"]

        self.render()
        self.update_time(init_h, init_m)

    def render(self) -> None:
        """Render the clock label, background image, and initial hands."""
        tk.Label(
            self.canvas,
            text="Remaining time",
            font=("Arial", 15),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).place(x=80, y=30)
        self.generate_bg()
        self.generate_clock_hands()

    def generate_bg(self) -> None:
        """Load and draw the clock-face background image."""
        self.image = tk.PhotoImage(
            file=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "graphics", "clock.gif"
            )
        )
        self.canvas.create_image(self.x, self.y, image=self.image)

    def generate_clock_hands(self) -> None:
        """Draw the initial hour and minute hands pointing to 12 o'clock."""
        self.sticks = []
        for i in range(2):
            store = self.canvas.create_line(
                self.x,
                self.y,
                self.x + self.length[i],
                self.y + self.length[i],
                width=self.width[i],
                fill=self.fill[i],
            )
            self.sticks.append(store)

    def update_time(self, h: int, m: int) -> None:
        """Move the clock hands to reflect the given elapsed time.

        Args:
            h: Elapsed hours from midnight (mapped to ``h * 5`` minute-units
                for the hour-hand angle calculation).
            m: Elapsed minutes within the current hour.
        """
        # h * 5 converts hours to clock-face units: the face has 60 positions
        # (one per minute), so each hour occupies 5 positions (60 / 12 = 5).
        now = (h * 5, m)
        # Changing Stick Coordinates
        for n, i in enumerate(now):
            cr = [
                self.x,
                self.y,
                self.length[n] * math.cos(math.radians(i * 6) - math.radians(90))
                + self.x,
                self.length[n] * math.sin(math.radians(i * 6) - math.radians(90))
                + self.y,
            ]
            self.canvas.coords(self.sticks[n], tuple(cr))  # type: ignore[index]
