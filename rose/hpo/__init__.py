"""
ROSE-HPO public API.

This module exposes the main classes and runtime entrypoints
so users can import them directly from `rose.hpo`.

Exported components:
- HPOLearner, HPOLearnerConfig: core HPO engine and configuration
- run_grid_hpo_task:    distributed grid search entrypoint
- run_random_hpo_task:  distributed random search entrypoint
- run_bayesian_hpo_task: distributed Bayesian optimization entrypoint
- run_genetic_hpo_task: distributed Genetic Algorithm entrypoint
- ROSEHPOManager:       unified high-level frontend for all HPO strategies
"""

from .hpo_learner import HPOLearner, HPOLearnerConfig
from .runtime_grid import run_grid_hpo_task
from .runtime_random import run_random_hpo_task
from .runtime_bayesian import run_bayesian_hpo_task
from .runtime_genetic import run_genetic_hpo_task
from .manager import ROSEHPOManager

__all__ = [
    "HPOLearner",
    "HPOLearnerConfig",
    "run_grid_hpo_task",
    "run_random_hpo_task",
    "run_bayesian_hpo_task",
    "run_genetic_hpo_task",
    "ROSEHPOManager",
]
