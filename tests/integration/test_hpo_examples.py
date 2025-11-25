import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]  # go up from tests/integration to repo root
EXAMPLES_DIR = ROOT / "examples" / "hpo"


def run_example(name: str):
    script = EXAMPLES_DIR / name
    assert script.exists(), f"Example script not found: {script}"

    result = subprocess.run(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    # For debugging if it fails
    print(result.stdout)

    assert result.returncode == 0, f"Example {name} exited with non-zero status"


@pytest.mark.parametrize("script_name", [
    "00-basic-grid.py",
    "01-random-search.py",
    "02-bayesian-search.py",
    "03-rose-runtime-grid.py",
    "04-rose-runtime-random.py",
    "05-rose-runtime-bayesian.py",
])
def test_examples_run_without_crash(script_name):
    run_example(script_name)
