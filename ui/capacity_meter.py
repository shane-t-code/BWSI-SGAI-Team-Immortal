import math
import tkinter as tk
from typing import Any

from gameplay.constants import UI_COLORS


class CapacityMeter:
    """Displays the ambulance capacity as a grid of colored unit squares.

    Each square represents one ambulance slot. Filled slots are shown in the
    accent (danger red) color; empty slots are shown in the panel background.

    Attributes:
        canvas: Tkinter canvas containing the unit squares and label.
        unit_size: Side length of each capacity square in pixels.
    """

    canvas: tk.Canvas
    unit_size: int
    _units: list[int]

    def __init__(self, root: Any, w: int, h: int, max_cap: int) -> None:
        """Create the capacity meter canvas and render all unit squares.

        Args:
            root: Tkinter parent widget.
            w: Window width in pixels (used to position the canvas).
            h: Window height in pixels (used to position the canvas).
            max_cap: Total number of ambulance slots to display.
        """
        self._canvas_w = math.floor(0.2 * w)
        self.canvas = tk.Canvas(
            root,
            width=self._canvas_w,
            height=math.floor(0.45 * h),
            bg=UI_COLORS["panel"],
            highlightthickness=0,
        )
        self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.4 * h))
        self._units = []
        self.unit_size = 40
        self.canvas.update()
        self.render(max_cap, self.unit_size)

    def render(self, max_cap: int, size: int) -> None:
        """Draw the capacity label and all unit squares on the canvas.

        Args:
            max_cap: Number of unit squares to create.
            size: Side length of each square in pixels.
        """
        tk.Label(
            self.canvas,
            text="Capacity",
            font=("Arial", 15),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).place(x=100, y=0)

        x = 3
        y = 50
        for _ in range(max_cap):
            self._units.append(self.create_unit(self.canvas, x, y, size))
            x += size * 3 // 2
            if (x + size * 3 // 2) > self._canvas_w:
                x = 3
                y += size * 3 // 2

    def update_fill(self, index: int) -> None:
        """Update the filled/empty state of the capacity squares.

        Clears all squares first, then fills slots ``0`` through ``index - 1``
        in the danger color. Pass ``index=0`` to show an empty ambulance.

        Args:
            index: Number of slots currently occupied.
        """
        for unit in self._units:
            self.canvas.itemconfig(unit, fill=UI_COLORS["panel"], outline="#555555")
        for i in range(index):
            self.canvas.itemconfig(
                self._units[i],
                fill=UI_COLORS["meter_filled"],
                outline=UI_COLORS["meter_filled"],
            )

    @staticmethod
    def create_unit(canvas: tk.Canvas, x: float, y: float, size: int) -> int:
        """Draw a single capacity square on the canvas and return its item ID.

        Args:
            canvas: Canvas to draw on.
            x: Left edge x-coordinate.
            y: Top edge y-coordinate.
            size: Side length of the square in pixels.

        Returns:
            Tkinter canvas item ID for the created rectangle.
        """
        return canvas.create_rectangle(
            x, y, x + size, y + size, fill=UI_COLORS["panel"], outline="#555555"
        )
