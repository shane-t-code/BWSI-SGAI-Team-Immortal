"""Tests for main.py CLI parsing.

Exercises build_parser() directly — no game loop, no Tkinter, no model weights.
Covers defaults, every flag, the train subcommand, and error paths.
"""

from __future__ import annotations

import pytest

from gameplay.constants import DATA_DIR
from main import _seed_all, build_parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse(args: list[str]):
    """Shorthand: parse a list of strings and return the Namespace."""
    return build_parser().parse_args(args)


def parse_err(args: list[str]) -> None:
    """Assert that parsing args exits with an error (argparse calls sys.exit)."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_all() -> None:
    """No args → human mode, log off, default data dir, no seed, no subcommand."""
    args = parse([])
    assert args.mode == "human"
    assert args.log is False
    assert args.data == DATA_DIR
    assert args.seed is None
    assert args.command is None


# ---------------------------------------------------------------------------
# -m / --mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["human", "machine", "human_cnn", "human_rl", "human_cnn_rl", "auto"],
)
def test_mode_all_valid(mode: str) -> None:
    """-m accepts every documented game mode."""
    assert parse(["-m", mode]).mode == mode


def test_mode_long_form() -> None:
    """--mode long form is accepted."""
    assert parse(["--mode", "auto"]).mode == "auto"


def test_mode_invalid_rejected() -> None:
    """An unrecognised mode causes a non-zero exit."""
    parse_err(["-m", "godmode"])


# ---------------------------------------------------------------------------
# -l / --log
# ---------------------------------------------------------------------------


def test_log_short() -> None:
    """-l sets log=True."""
    assert parse(["-l"]).log is True


def test_log_long() -> None:
    """--log sets log=True."""
    assert parse(["--log"]).log is True


def test_log_default_false() -> None:
    """log is False when flag is absent."""
    assert parse([]).log is False


# ---------------------------------------------------------------------------
# -s / --seed
# ---------------------------------------------------------------------------


def test_seed_short() -> None:
    """-s accepts an integer seed."""
    assert parse(["-s", "42"]).seed == 42


def test_seed_long() -> None:
    """--seed accepts an integer seed."""
    assert parse(["--seed", "7"]).seed == 7


def test_seed_default_none() -> None:
    """seed is None when omitted."""
    assert parse([]).seed is None


def test_seed_non_integer_rejected() -> None:
    """A non-integer seed causes a non-zero exit."""
    parse_err(["-s", "notanint"])


# ---------------------------------------------------------------------------
# -d / --data
# ---------------------------------------------------------------------------


def test_data_short() -> None:
    """-d overrides the data directory."""
    assert parse(["-d", "/custom/path"]).data == "/custom/path"


def test_data_long() -> None:
    """--data long form is accepted."""
    assert parse(["--data", "/other/path"]).data == "/other/path"


# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------


def test_train_type_rl() -> None:
    """`train --type rl` sets command=train and type=rl."""
    args = parse(["train", "--type", "rl"])
    assert args.command == "train"
    assert args.type == "rl"


def test_train_type_cnn() -> None:
    """`train --type cnn` sets command=train and type=cnn."""
    args = parse(["train", "--type", "cnn"])
    assert args.command == "train"
    assert args.type == "cnn"


def test_train_missing_type_rejected() -> None:
    """`train` without --type causes a non-zero exit."""
    parse_err(["train"])


def test_train_invalid_type_rejected() -> None:
    """`train --type transformer` causes a non-zero exit."""
    parse_err(["train", "--type", "transformer"])


def test_train_epochs_flag() -> None:
    """`train -e 500` sets epochs=500."""
    assert parse(["train", "--type", "rl", "-e", "500"]).epochs == 500


def test_train_epochs_default_none() -> None:
    """epochs defaults to None (caller applies model-specific default)."""
    assert parse(["train", "--type", "rl"]).epochs is None


def test_train_seed_flag() -> None:
    """`train -s 99` sets seed=99."""
    assert parse(["train", "--type", "cnn", "-s", "99"]).seed == 99


def test_train_data_flag() -> None:
    """`train -d /data` sets data directory."""
    assert parse(["train", "--type", "rl", "-d", "/data"]).data == "/data"


def test_train_output_flag() -> None:
    """`train -o /weights.pt` sets output path."""
    assert parse(["train", "--type", "rl", "-o", "/weights.pt"]).output == "/weights.pt"


def test_train_output_default_none() -> None:
    """output defaults to None (caller applies model-specific default path)."""
    assert parse(["train", "--type", "rl"]).output is None


# ---------------------------------------------------------------------------
# _seed_all
# ---------------------------------------------------------------------------


def test_seed_all_none_is_noop() -> None:
    """_seed_all(None) must not raise."""
    _seed_all(None)


def test_seed_all_integer_runs() -> None:
    """_seed_all with an integer seed must not raise."""
    _seed_all(42)
    _seed_all(0)
