from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend  # type: ignore

from .hpo_learner import HPOLearner
from .runtime import run_grid_search_distributed


# ===========================================================================
# INTERNAL ASYNC WRAPPER
# ===========================================================================
async def _run_grid_async(
    hpo: HPOLearner,
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Internal async helper for distributed GRID search.

    Steps:
      1. Create AsyncFlow backend (ProcessPoolExecutor)
      2. Create WorkflowEngine
      3. Run distributed grid search
      4. Shutdown engine safely
    """

    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend=backend)

    try:
        result = await run_grid_search_distributed(
            asyncflow=asyncflow,
            hpo=hpo,
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
        )
    finally:
        await asyncflow.shutdown()

    return result


# ===========================================================================
# PUBLIC SYNCHRONOUS ENTRYPOINT
# ===========================================================================
def run_grid_hpo_task(
    hpo: HPOLearner,
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Synchronous entrypoint for GRID search.

    Example usage:
        result = run_grid_hpo_task(
            hpo,
            checkpoint_path="grid_history.json",
            checkpoint_freq=5
        )

    Parameters
    ----------
    hpo : HPOLearner
        Configured learner with param_grid and objective_fn.

    checkpoint_path : str or None
        If provided, the distributed evaluator will save HPO history
        to this JSON file every `checkpoint_freq` successful evaluations.

    checkpoint_freq : int
        Save a checkpoint every N successful evaluations.
        If <= 0, checkpointing is disabled.
    """

    return asyncio.run(
        _run_grid_async(
            hpo=hpo,
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
        )
    )
