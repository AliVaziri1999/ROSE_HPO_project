import runpy
from pathlib import Path


def test_rose_runtime_bayesian_runs():
    """
    Smoke test: the Bayesian-optimization HPO example should run to completion
    under the ROSE runtime without raising exceptions.
    """
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "examples" / "hpo" / "05-rose-runtime-bayesian.py"

    runpy.run_path(str(script), run_name="__main__")
