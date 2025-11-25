from __future__ import annotations

from typing import Any, Dict

from .hpo_learner import HPOLearner


def run_bayesian_hpo_task(
    hpo: HPOLearner,
    n_init: int = 5,
    n_iter: int = 15,
    kappa: float = 2.0,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Local Bayesian Optimization wrapper.
    """
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
