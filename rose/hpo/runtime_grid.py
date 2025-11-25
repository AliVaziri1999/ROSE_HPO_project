from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend  # type: ignore

from .hpo_learner import HPOLearner
from .runtime import run_grid_search_distributed


async def _run_grid_async(hpo: HPOLearner) -> Dict[str, Any]:
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
        result = await run_grid_search_distributed(asyncflow, hpo)
    finally:
        await asyncflow.shutdown()

    return result


def run_grid_hpo_task(hpo: HPOLearner) -> Dict[str, Any]:
    """
    Synchronous entrypoint for GRID search.

    This is what the example scripts call. Internally it spins up
    the async workflow and runs the distributed execution.
    """
    return asyncio.run(_run_grid_async(hpo))
