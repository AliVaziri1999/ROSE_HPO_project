from __future__ import annotations

from typing import Any, Callable, Dict

from .hpo_learner import HPOLearner, HPOLearnerConfig
from .runtime_grid import run_grid_hpo_task
from .runtime_random import run_random_hpo_task
from .runtime_bayesian import run_bayesian_hpo_task
from .runtime_genetic import run_genetic_hpo_task


class ROSEHPOManager:
    """
    Centralized, user-facing HPO manager.

    This class is meant to be the *single entrypoint* for the rest of the code:
      - The user (or example scripts) only needs to choose:
          * a strategy: "grid", "random", "bayesian", or "genetic"
          * a hyperparameter search space (param_grid)
          * an objective function: params -> score
      - The manager then:
          * builds an HPOLearnerConfig
          * constructs an HPOLearner
          * calls the correct runtime helper based on the chosen strategy

    All strategies return the same kind of result dictionary so downstream code
    does not have to care how the search was done.
    """

    def __init__(
        self,
        strategy: str,
        param_grid: Dict[str, Any],
        maximize: bool = False,
        # RANDOM search parameters
        n_samples: int = 20,
        rng_seed: int = 0,
        # BAYESIAN search parameters
        n_init: int = 5,
        n_iter: int = 15,
        kappa: float = 2.0,
        # GENETIC ALGORITHM parameters
        population_size: int = 10,
        n_generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        tournament_size: int = 3,
        elitism: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        strategy :
            Which HPO strategy to use.
            Supported values:
                - "grid"
                - "random"
                - "bayesian"
                - "genetic"
        param_grid :
            Dict mapping hyperparameter name (str) -> list of candidate values.
            All strategies read their search space from here.
        maximize :
            If True, the objective function is treated as "higher is better"
            (e.g., accuracy). If False, "lower is better" (e.g., loss).

        n_samples :
            RANDOM search only. Number of randomly sampled configurations.
        rng_seed :
            RANDOM / BAYESIAN / GENETIC use this seed for reproducibility.

        n_init :
            BAYESIAN search only. Number of pure random points to evaluate
            before building the surrogate model.
        n_iter :
            BAYESIAN search only. Number of Bayesian optimization iterations
            after the initial random points.
        kappa :
            BAYESIAN search only. Exploration–exploitation trade-off parameter
            for the UCB acquisition function.

        population_size :
            GENETIC only. Number of individuals (hyperparameter configurations)
            in each generation.
        n_generations :
            GENETIC only. How many generations to evolve the population.
        mutation_rate :
            GENETIC only. Probability of mutating each hyperparameter value
            during mutation.
        crossover_rate :
            GENETIC only. Probability that crossover will be applied when
            creating offspring.
        tournament_size :
            GENETIC only. Number of individuals participating in each
            tournament selection.
        elitism :
            GENETIC only. How many of the top individuals are copied directly
            to the next generation (without any changes).
        """
        self.strategy = strategy
        self.param_grid = param_grid
        self.maximize = maximize

        # RANDOM
        self.n_samples = n_samples
        self.rng_seed = rng_seed

        # BAYESIAN
        self.n_init = n_init
        self.n_iter = n_iter
        self.kappa = kappa

        # GENETIC
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def _build_learner(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
    ) -> HPOLearner:
        """
        Small helper that wraps the two-step pattern:
          1) build an HPOLearnerConfig
          2) construct an HPOLearner with the user's objective function
        """
        cfg = HPOLearnerConfig(
            param_grid=self.param_grid,
            maximize=self.maximize,
        )
        return HPOLearner(objective_fn=objective_fn, config=cfg)

    def run(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run HPO with the chosen strategy.

        Parameters
        ----------
        objective_fn :
            A function that:
                - takes a single argument: params (dict of hyperparameters)
                - returns a scalar score (float)

            Example shape:
                def objective_fn(params: Dict[str, Any]) -> float:
                    model = build_model(**params)
                    score = train_and_evaluate(model)
                    return score

        Returns
        -------
        dict with keys:
            - "best_params": dict
            - "best_score": float
            - "history": list of {"params": ..., "score": ...}
        """
        # Build a fresh learner for this run. The learner manages:
        #   - how we iterate over or sample hyperparameter configurations
        #   - the running history of all trials
        hpo = self._build_learner(objective_fn)

        # --- GRID SEARCH ----------------------------------------------------
        if self.strategy == "grid":
            return run_grid_hpo_task(hpo)

        # --- RANDOM SEARCH --------------------------------------------------
        if self.strategy == "random":
            return run_random_hpo_task(
                hpo,
                n_samples=self.n_samples,
                rng_seed=self.rng_seed,
            )

        # --- BAYESIAN OPTIMIZATION -----------------------------------------
        if self.strategy == "bayesian":
            return run_bayesian_hpo_task(
                hpo,
                n_init=self.n_init,
                n_iter=self.n_iter,
                kappa=self.kappa,
                rng_seed=self.rng_seed,
            )

        # --- GENETIC ALGORITHM ---------------------------------------------
        if self.strategy == "genetic":
            return run_genetic_hpo_task(
                hpo,
                population_size=self.population_size,
                n_generations=self.n_generations,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                tournament_size=self.tournament_size,
                elitism=self.elitism,
                rng_seed=self.rng_seed,
            )

        # If we reach here, the strategy name is unknown.
        raise ValueError(f"Unknown HPO strategy: {self.strategy!r}")
