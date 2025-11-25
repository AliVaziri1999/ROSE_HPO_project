from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend  # type: ignore

from .hpo_learner import HPOLearner
from .runtime import run_genetic_search_distributed


async def _run_ga_async(
    hpo: HPOLearner,
    population_size: int,
    n_generations: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    elitism: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """
    Internal async helper for distributed Genetic Algorithm HPO.
    """
    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend=backend)

    try:
        result = await run_genetic_search_distributed(
            asyncflow=asyncflow,
            hpo=hpo,
            population_size=population_size,
            n_generations=n_generations,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            rng_seed=rng_seed,
        )
    finally:
        await asyncflow.shutdown()

    return result


def run_genetic_hpo_task(
    hpo: HPOLearner,
    population_size: int = 20,
    n_generations: int = 30,
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    elitism: int = 1,
    rng_seed: int = 0,
    distributed: bool = False,
) -> Dict[str, Any]:
    """
    ROSE runtime entrypoint for Genetic Algorithm HPO.

    If distributed=False (default):
        - use the local HPOLearner.genetic_search() implementation.

    If distributed=True:
        - use the ROSE / asyncflow-based distributed evaluation
          defined in runtime.run_genetic_search_distributed().
    """
    if not distributed:
        # Local (non-distributed) path
        best_params, best_score, history = hpo.genetic_search(
            population_size=population_size,
            n_generations=n_generations,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            rng_seed=rng_seed,
        )
        return {
            "best_params": best_params,
            "best_score": best_score,
            "history": history,
        }

    # Distributed path using AsyncFlow
    return asyncio.run(
        _run_ga_async(
            hpo=hpo,
            population_size=population_size,
            n_generations=n_generations,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            rng_seed=rng_seed,
        )
    )
