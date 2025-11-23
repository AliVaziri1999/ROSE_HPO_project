import runpy
from pathlib import Path


def test_rose_runtime_random_runs():
    """
    Smoke test: the random-search HPO example should run to completion
    under the ROSE runtime without raising exceptions.
    """
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "examples" / "hpo" / "04-rose-runtime-random.py"

    # run the example as if it were __main__
    runpy.run_path(str(script), run_name="__main__")
