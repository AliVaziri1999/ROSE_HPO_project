from __future__ import annotations

from typing import Any, Dict

from .hpo_learner import HPOLearner


def run_genetic_hpo_task(
    hpo: HPOLearner,
    population_size: int = 20,
    n_generations: int = 30,
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    elitism: int = 1,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Local Genetic Algorithm HPO wrapper.

    (We could later upgrade this to a distributed version by reusing
    `_evaluate_configs_distributed` for parallel fitness evaluation.)
    """
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
