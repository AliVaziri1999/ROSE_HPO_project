from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Project root: tests/integration/ -> tests -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "hpo"

# Only what the assignment needs: grid, random, bayesian
EXAMPLE_SCRIPTS = [
    # Local scripts
    EXAMPLES_DIR / "local" / "00-local-grid.py",
    EXAMPLES_DIR / "local" / "01-local-random.py",
    EXAMPLES_DIR / "local" / "02-local-bayesian.py",

    # Runtime scripts
    EXAMPLES_DIR / "runtime" / "00-runtime-grid.py",
    EXAMPLES_DIR / "runtime" / "01-runtime-random.py",
    EXAMPLES_DIR / "runtime" / "02-runtime-bayesian.py",
]

# Minimal argument sets for each script so they can run
BASE_ARGS = [
    "--learning_rate", "0.001",
    "--num_layers", "1",
    "--hidden_units", "64",
    "--batch_size", "64",
    "--l2_reg", "0.0",
]

ARGS_BY_SCRIPT = {
    # local examples
    "00-local-grid.py": BASE_ARGS,
    "01-local-random.py": BASE_ARGS + [
        "--n_samples", "3",
        "--rng_seed", "0",
    ],
    "02-local-bayesian.py": BASE_ARGS + [
        "--n_init", "2",
        "--n_iter", "3",
        "--kappa", "2.0",
        "--rng_seed", "0",
    ],

    # runtime examples
    "00-runtime-grid.py": BASE_ARGS + [
        "--epochs", "1",
    ],
    # these already have defaults and were passing without args
    "01-runtime-random.py": [],
    "02-runtime-bayesian.py": [],
}


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS,
                         ids=[p.name for p in EXAMPLE_SCRIPTS])
def test_examples_run_without_crash(script_path: Path) -> None:
    """
    Each HPO example script (local and runtime, grid/random/bayesian)
    should exist in Ali's structure and run without crashing.
    """
    assert script_path.exists(), f"Example script not found: {script_path}"

    extra_args = ARGS_BY_SCRIPT.get(script_path.name, [])

    result = subprocess.run(
        [sys.executable, str(script_path), *extra_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, (
        f"Script {script_path} exited with code {result.returncode}\n"
        f"=== STDOUT ===\n{result.stdout}\n"
        f"=== STDERR ===\n{result.stderr}\n"
    )
