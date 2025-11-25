from __future__ import annotations

from typing import Any, Dict

from .hpo_learner import HPOLearner


def run_random_hpo_task(
    hpo: HPOLearner,
    n_samples: int = 20,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Local (non-distributed) RANDOM search wrapper.

    Returns a dict with:
        - best_params
        - best_score
        - history
    """
    best_params, best_score, history = hpo.random_search(
        n_samples=n_samples,
        rng_seed=rng_seed,
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history,
    }
