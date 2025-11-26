from __future__ import annotations

import sys
import pytest

from rose.hpo import bootstrap_hpo


def _run_parse(argv: list[str]):
    """
    Helper: set sys.argv so that bootstrap_hpo.parse_args()
    reads our fake CLI arguments, then restore sys.argv.
    """
    old_argv = sys.argv
    sys.argv = ["bootstrap_hpo.py"] + argv
    try:
        args = bootstrap_hpo.parse_args()
    finally:
        sys.argv = old_argv
    return args


def test_parse_args_defaults():
    """
    Check that parse_args() works with no extra CLI args and
    returns a Namespace with expected basic attributes.
    This matches your current CLI that uses defaults.
    """
    args = _run_parse([])

    # Strategy should be one of allowed ones and default to "grid"
    assert args.strategy in ["grid", "random", "bayesian", "ga"]
    # We know default is "grid" in your code
    assert args.strategy == "grid"

    # Shared search space exists and is list-like
    assert isinstance(args.learning_rate, list)
    assert isinstance(args.num_layers, list)
    assert isinstance(args.hidden_units, list)
    assert isinstance(args.batch_size, list)
    assert isinstance(args.l2_reg, list)

    # Some global knobs exist
    assert isinstance(args.epochs, int)
    assert hasattr(args, "no_normalize")


def test_parse_args_invalid_strategy():
    """
    If user passes an invalid strategy, argparse should raise SystemExit,
    which is the standard behavior and also what we expect for the project.
    """
    with pytest.raises(SystemExit):
        _run_parse(["--strategy", "invalid_strategy"])
