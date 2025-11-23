from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from radical.asyncflow import WorkflowEngine

from .hpo_learner import HPOLearner


async def run_grid_search_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
) -> Dict[str, Any]:
    """
    Run a grid search where **each hyperparameter configuration**
    is evaluated as its own ROSE / asyncflow task.

    Parameters
    ----------
    asyncflow :
        An initialized WorkflowEngine instance (using ConcurrentExecutionBackend).
    hpo :
        HPOLearner configured with:
          - objective_fn: params -> score (float)
          - config.param_grid: dict of hyperparameter lists

    Returns
    -------
    dict with keys:
        - "best_params": dict
        - "best_score": float
        - "history": list of {"params": ..., "score": ...}
    """

    # Collect all parameter combinations from the learner's grid
    param_list: List[Dict[str, Any]] = list(hpo._iter_param_combinations())
    local_history: List[Dict[str, Any]] = []

    # One async function per configuration – this becomes a ROSE task
    async def eval_one(params: Dict[str, Any]) -> Dict[str, Any]:
        # Runs inside the worker process. We call the user's objective
        # synchronously here; parallelism comes from the ProcessPool backend.
        score = hpo.objective_fn(params)
        # Force plain float for easier JSON / logging
        return {"params": params, "score": float(score)}

    # Wrap eval_one as an asyncflow function task
    eval_task = asyncflow.function_task(eval_one)

    # Submit **one task per configuration**
    futures = [eval_task(p) for p in param_list]

    # Wait for all tasks to finish
    results: List[Dict[str, Any]] = await asyncio.gather(*futures)

    # Aggregate and find best
    best_params: Dict[str, Any] | None = None
    best_score: float | None = None

    for rec in results:
        local_history.append(rec)
        score = rec["score"]

        if best_score is None:
            best_score = score
            best_params = rec["params"]
            continue

        if hpo.config.maximize:
            if score > best_score:
                best_score = score
                best_params = rec["params"]
        else:
            if score < best_score:
                best_score = score
                best_params = rec["params"]

    # Extend the learner's history so you still have all trials there
    hpo.history.extend(local_history)

    # Defensive check (should not happen if grid is non-empty)
    if best_params is None or best_score is None:
        raise ValueError("param_grid is empty – nothing to search over.")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": hpo.history,
    }
