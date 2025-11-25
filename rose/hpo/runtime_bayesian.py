from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend  # type: ignore

from .hpo_learner import HPOLearner
from .runtime import run_bayesian_search_distributed


async def _run_bayesian_async(
    hpo: HPOLearner,
    n_init: int,
    n_iter: int,
    kappa: float,
    rng_seed: int,
) -> Dict[str, Any]:
    """
    Internal async helper for distributed Bayesian Optimization.
    """
    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend=backend)

    try:
        result = await run_bayesian_search_distributed(
            asyncflow=asyncflow,
            hpo=hpo,
            n_init=n_init,
            n_iter=n_iter,
            kappa=kappa,
            rng_seed=rng_seed,
        )
    finally:
        await asyncflow.shutdown()

    return result


def run_bayesian_hpo_task(
    hpo: HPOLearner,
    n_init: int = 5,
    n_iter: int = 15,
    kappa: float = 2.0,
    rng_seed: int = 0,
    distributed: bool = False,
) -> Dict[str, Any]:
    """
    ROSE runtime entrypoint for BAYESIAN optimization.

    If distributed=False (default):
        - use the local HPOLearner.bayesian_search() implementation.

    If distributed=True:
        - use the ROSE / asyncflow-based distributed evaluation
          defined in runtime.run_bayesian_search_distributed().
    """
    if not distributed:
        # Local (non-distributed) path
        best_params, best_score, history = hpo.bayesian_search(
            n_init=n_init,
            n_iter=n_iter,
            kappa=kappa,
            rng_seed=rng_seed,
        )
        return {
            "best_params": best_params,
            "best_score": best_score,
            "history": history,
        }

    # Distributed path using AsyncFlow
    return asyncio.run(
        _run_bayesian_async(
            hpo=hpo,
            n_init=n_init,
            n_iter=n_iter,
            kappa=kappa,
            rng_seed=rng_seed,
        )
    )
