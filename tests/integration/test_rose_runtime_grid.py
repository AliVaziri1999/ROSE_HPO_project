import subprocess
import sys
from pathlib import Path

def test_rose_runtime_hpo_runs():
    script = Path("examples/hpo/03-rose-runtime-grid.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "HPO via ROSE runtime (grid search)" in result.stdout
