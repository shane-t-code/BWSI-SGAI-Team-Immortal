"""Tests that required UI asset files are present at expected paths."""

import os

_GRAPHICS_DIR = os.path.join(os.path.dirname(__file__), "..", "ui", "graphics")


def test_bwsi_logo_exists() -> None:
    """logo.png (BWSI logo) must exist in ui/graphics/."""
    path = os.path.join(_GRAPHICS_DIR, "logo.png")
    assert os.path.isfile(path), f"BWSI logo missing: {path}"


def test_game_logo_exists() -> None:
    """logo2.png (game logo) must exist in ui/graphics/."""
    path = os.path.join(_GRAPHICS_DIR, "logo2.png")
    assert os.path.isfile(path), f"Game logo missing: {path}"
