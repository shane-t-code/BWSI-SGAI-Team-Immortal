import math
import tkinter as tk
from os.path import join
from typing import Any
from PIL import ImageTk, Image

from gameplay.constants import UI_COLORS


class GameViewer:
    """Central photo panel — displays the current humanoid image and end-game score.

    Attributes:
        canvas: Tkinter canvas that holds the humanoid photo.
        photo: Currently displayed :class:`~PIL.ImageTk.PhotoImage`, or
            ``None`` if no photo is loaded. Held as an instance attribute to
            prevent garbage collection.
    """

    canvas: tk.Canvas
    photo: Any | None  # ImageTk.PhotoImage | None

    def __init__(self, root: Any, w: int, h: int, data_fp: str, humanoid: Any) -> None:
        """Create the game viewer canvas and display the first humanoid.

        Args:
            root: Tkinter parent widget.
            w: Window width in pixels (used to size and position the canvas).
            h: Window height in pixels (used to size the canvas).
            data_fp: Root path for image files.
            humanoid: Initial :class:`~gameplay.humanoid.Humanoid` whose image
                is displayed on creation.
        """
        self.canvas = tk.Canvas(
            root,
            width=math.floor(0.5 * w),
            height=math.floor(0.75 * h),
            bg=UI_COLORS["panel"],
            highlightthickness=0,
        )
        self.canvas.place(x=300, y=100)
        self.canvas.update()

        self.photo = None
        self.create_photo(join(data_fp, humanoid.fp))

    def delete_photo(self, event: Any = None) -> None:
        """Remove the current photo from the canvas.

        Args:
            event: Tkinter event passed when called as a key binding, or
                ``None`` when called directly.
        """
        self.canvas.delete("photo")

    def create_photo(self, fp: str) -> None:
        """Load an image from disk and display it on the canvas.

        Replaces any existing photo. Holds a reference in ``self.photo``
        to prevent garbage collection.

        Args:
            fp: Path to the image file.
        """
        self.canvas.delete("photo")
        self.photo = self._get_photo(
            fp, self.canvas.winfo_width(), self.canvas.winfo_height()
        )
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="photo")

    def display_score(
        self, score: dict[str, int | float], end_reason: str = ""
    ) -> None:
        """Render the final score over the canvas.

        Args:
            score: Dict with ``"killed"`` and ``"saved"`` integer counts and
                ``"saved_value"`` float (total value of saved humanoids),
                as returned by :meth:`~gameplay.scorekeeper.ScoreKeeper.get_score`.
            end_reason: Optional end-of-game reason string (e.g. ``"Time's Up"``
                or ``"The City Is Empty"``). Displayed as a bold header if provided.
        """
        if end_reason:
            tk.Label(
                self.canvas,
                text=end_reason,
                font=("Arial", 20, "bold"),
                bg=UI_COLORS["panel"],
                fg=UI_COLORS["accent"],
            ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="FINAL SCORE",
            font=("Arial", 30, "bold"),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="Killed {}".format(score["killed"]),
            font=("Arial", 18),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["accent"],
        ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="Saved {}".format(score["saved"]),
            font=("Arial", 18),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.NW)
        tk.Label(
            self.canvas,
            text="Saved Value {0:.4f}".format(score["saved_value"]),
            font=("Arial", 18),
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["fg"],
        ).pack(anchor=tk.NW)

    @staticmethod
    def _get_photo(img_path: str, w: int, h: int) -> Any:
        """Load, resize, and return a Tkinter-compatible photo image.

        Args:
            img_path: Path to the image file.
            w: Target width in pixels.
            h: Target height in pixels.

        Returns:
            An :class:`~PIL.ImageTk.PhotoImage` resized to ``(w, h)``.
        """
        with Image.open(img_path) as img:
            # Convert inside the with block so the file handle is still open
            # during resize, then construct the PhotoImage before closing.
            tk_img = ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))  # type: ignore[attr-defined]
        return tk_img
