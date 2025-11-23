"""
ROSE-HPO public API.

This exports:
- HPOLearner: the core local (synchronous) HPO engine
- HPOLearnerConfig: config object for grid/random/BO
- run_grid_hpo_task: ROSE runtime entrypoint
- run_random_hpo_task: ROSE runtime entrypoint
- run_bayesian_hpo_task: ROSE runtime entrypoint
- ROSEHPOManager: unified frontend for the assignment
"""

from .hpo_learner import HPOLearner, HPOLearnerConfig
from .runtime_grid import run_grid_hpo_task
from .runtime_random import run_random_hpo_task
from .runtime_bayesian import run_bayesian_hpo_task
from .manager import ROSEHPOManager

__all__ = [
    "HPOLearner",
    "HPOLearnerConfig",
    "run_grid_hpo_task",
    "run_random_hpo_task",
    "run_bayesian_hpo_task",
    "ROSEHPOManager",
]
