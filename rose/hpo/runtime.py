from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import numpy as np 

from radical.asyncflow import WorkflowEngine

from .hpo_learner import HPOLearner


# ---------------------------------------------------------------------------
# Internal helper: run a LIST of hyperparameter configs as ROSE tasks
# ---------------------------------------------------------------------------

async def _evaluate_configs_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    param_list: List[Dict[str, Any]],
    context_label: str = "HPO",
) -> Dict[str, Any]:
    """
    Submit a list of hyperparameter configurations to ROSE / asyncflow,
    wait for all of them to finish, and return the best one.

    This helper is shared by grid search, random search, and can also be
    reused by Bayesian / GA runtimes.

    Parameters
    ----------
    asyncflow :
        An initialized WorkflowEngine instance (usually with ConcurrentExecutionBackend).
    hpo :
        HPOLearner instance. Must provide:
          - objective_fn(params) -> float
          - config.maximize: bool (True = higher score is better)
          - history: list to be extended with all evaluated trials
    param_list :
        List of hyperparameter configurations (each a dict) to evaluate.
    context_label :
        Small string used in log messages, just to identify which strategy
        is calling this helper (e.g., "GRID", "RANDOM", "BAYESIAN", "GA").

    Returns
    -------
    dict with keys:
        - "best_params": dict
        - "best_score": float
        - "history": list of {"params": ..., "score": ...}
    """
    if not param_list:
        raise ValueError(
            f"{context_label} received an empty param_list – nothing to evaluate."
        )

    print(f"[{context_label}] Submitting {len(param_list)} configurations to ROSE...")

    local_history: List[Dict[str, Any]] = []

    # This function is what each worker process actually runs.
    async def eval_one(params: Dict[str, Any]) -> Dict[str, Any]:
        # Note: objective_fn is usually a *blocking* function that trains
        # a model and returns a scalar score. Parallelism comes from the
        # underlying ProcessPool in asyncflow, not from async logic inside
        # this function.
        score = hpo.objective_fn(params)
        # Cast to plain float to keep JSON / logging clean.
        return {"params": params, "score": float(score)}

    # Wrap eval_one into an asyncflow "function task" so each call becomes
    # a distributed / parallel task.
    eval_task = asyncflow.function_task(eval_one)

    # One task per configuration
    futures = [eval_task(p) for p in param_list]

    # Wait for all tasks to finish
    results: List[Dict[str, Any]] = await asyncio.gather(*futures)

    # Find the best configuration according to the "maximize" flag
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

    # Extend the learner's history so we keep a global record of all trials
    hpo.history.extend(local_history)

    if best_params is None or best_score is None:
        raise ValueError(f"[{context_label}] No valid results – unexpected.")

    print(
        f"[{context_label}] Best score: {best_score:.4f} "
        f"with params: {best_params}"
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": hpo.history,
    }


# ---------------------------------------------------------------------------
# GRID SEARCH – one task per configuration
# ---------------------------------------------------------------------------

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

    # Delegate to the shared distributed evaluator
    return await _evaluate_configs_distributed(
        asyncflow=asyncflow,
        hpo=hpo,
        param_list=param_list,
        context_label="GRID",
    )


# ---------------------------------------------------------------------------
# RANDOM SEARCH – sample n configurations from the grid, then distribute
# ---------------------------------------------------------------------------

async def run_random_search_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    n_samples: int,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Distributed RANDOM search using ROSE / asyncflow.

    Instead of enumerating the full Cartesian grid, we:
      - sample `n_samples` hyperparameter configurations *with replacement*
        from `hpo.config.param_grid`
      - evaluate each sampled configuration as its OWN asyncflow task.

    Parameters
    ----------
    asyncflow :
        An initialized WorkflowEngine instance.
    hpo :
        HPOLearner configured with an objective_fn and param_grid.
    n_samples :
        How many random configurations to evaluate.
    rng_seed :
        Seed for the random number generator (for reproducibility).

    Returns
    -------
    dict with keys:
        - "best_params": dict
        - "best_score": float
        - "history": list of {"params": ..., "score": ...}
          (this includes previous hpo.history entries plus these trials)
    """
    param_grid = hpo.config.param_grid
    if not param_grid:
        raise ValueError("param_grid is empty - nothing to sample from.")

    rng = np.random.default_rng(rng_seed)
    keys = list(param_grid.keys())
    value_arrays = [np.array(param_grid[k]) for k in keys]

    # Sample n_samples random configs (with replacement)
    sampled_params: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        cfg = {
            k: value_arrays[i][rng.integers(0, len(value_arrays[i]))]
            for i, k in enumerate(keys)
        }
        sampled_params.append(cfg)

    # Delegate to the shared evaluator
    return await _evaluate_configs_distributed(
        asyncflow=asyncflow,
        hpo=hpo,
        param_list=sampled_params,
        context_label="RANDOM",
    )
