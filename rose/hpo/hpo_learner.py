from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
from pydantic import BaseModel
import itertools


class HPOLearnerConfig(BaseModel):
    """
    Simple configuration object for HPO runs.

    Attributes
    ----------
    param_grid :
        Dictionary that maps a hyperparameter name to a list of
        candidate values, e.g.:

        {
            "length_scale": [0.1, 0.5, 1.0],
            "noise_level": [0.01, 0.1]
        }

    maximize :
        If True, the learner will choose the configuration with the
        *largest* score. If False, it will choose the *smallest* score.
        For example:
        - loss / RMSE  -> maximize = False
        - accuracy     -> maximize = True
    """

    param_grid: Dict[str, List[Any]]
    maximize: bool = False


class HPOLearner:
    """
    Very small, local HPO helper.

    This class is the local (non-distributed) engine that all our examples
    use underneath. It is intentionally simple:

    - no async, no WorkflowEngine, no HPC
    - just loops or samples over configurations in `param_grid`
    - calls a user-provided objective function for each config
    - tracks everything in `self.history`
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        config: HPOLearnerConfig,
    ) -> None:
        """
        Parameters
        ----------
        objective_fn :
            A function that receives a dict of hyperparameters and
            returns a single numeric score (float).
            Example signature:
                def objective(params: dict) -> float:
                    ...
                    return rmse   # or accuracy, etc.

        config :
            HPOLearnerConfig with param_grid and maximize flag.
        """
        self.objective_fn = objective_fn
        self.config = config

        # Store all tried configs + scores:
        #   [{"params": {...}, "score": float}, ...]
        self.history: List[Dict[str, Any]] = []

    def _iter_param_combinations(self):
        """
        Generate all combinations from param_grid.

        This is the "Cartesian product" over the lists in param_grid.
        """
        keys = list(self.config.param_grid.keys())
        value_lists = [self.config.param_grid[k] for k in keys]

        for combo in itertools.product(*value_lists):
            params = {k: v for k, v in zip(keys, combo)}
            yield params

    # ----------------------------------------------------------------------
    # GRID SEARCH
    # ----------------------------------------------------------------------
    def grid_search(self) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Run a basic grid search over all combinations in param_grid.

        Returns
        -------
        best_params :
            Dict with best hyperparameter values.

        best_score :
            Score for the best configuration (float).

        history :
            List of dicts, each like:
                {
                    "params": {...},
                    "score": float
                }
        """
        best_params: Dict[str, Any] | None = None
        best_score: float | None = None

        for params in self._iter_param_combinations():
            score = self.objective_fn(params)

            self.history.append({"params": params, "score": score})

            if best_score is None:
                best_score = score
                best_params = params
                continue

            if self.config.maximize:
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                if score < best_score:
                    best_score = score
                    best_params = params

        # Defensive: should not happen if grid is non–empty
        if best_params is None or best_score is None:
            raise ValueError("param_grid is empty – nothing to search over.")

        return best_params, best_score, self.history

    # ----------------------------------------------------------------------
    # RANDOM SEARCH
    # ----------------------------------------------------------------------
    def random_search(
        self,
        n_samples: int,
        rng_seed: int = 0,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Run basic random search.

        Parameters
        ----------
        n_samples : int
            Number of random configurations to sample.
        rng_seed : int
            Seed for reproducibility.

        Returns
        -------
        best_params : dict
        best_score : float
        history : list(dict)
        """
        import numpy as np

        rng = np.random.default_rng(rng_seed)

        # Build list of keys
        keys = list(self.config.param_grid.keys())

        # Preconvert each key to a numpy array for fast random indexing
        value_arrays = [np.array(self.config.param_grid[k]) for k in keys]

        best_params: Dict[str, Any] | None = None
        best_score: float | None = None

        for _ in range(n_samples):
            # Choose random value for each hyperparameter
            params = {
                k: value_arrays[i][rng.integers(0, len(value_arrays[i]))]
                for i, k in enumerate(keys)
            }

            score = self.objective_fn(params)
            self.history.append({"params": params, "score": score})

            if best_score is None:
                best_score = score
                best_params = params
                continue

            if self.config.maximize:
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                if score < best_score:
                    best_score = score
                    best_params = params

        return best_params, best_score, self.history

    # ----------------------------------------------------------------------
    # BAYESIAN OPTIMIZATION
    # ----------------------------------------------------------------------
    def bayesian_search(
        self,
        n_init: int = 5,
        n_iter: int = 15,
        kappa: float = 2.0,
        rng_seed: int = 0,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Very simple Bayesian Optimization over a discrete grid.

        We:
        - evaluate `n_init` random configs
        - then repeat `n_iter` times:
            * fit a GP surrogate on (params -> score)
            * for all unseen configs, compute an Upper/Lower Confidence Bound
              acquisition function
            * pick the best according to that acquisition

        Uses:
        - minimize mode: acq = mu - kappa * sigma  (Lower Confidence Bound)
        - maximize mode: acq = -(mu + kappa * sigma) (so we still take argmin)

        Returns
        -------
        best_params : dict
        best_score : float
        history : list(dict)
        """
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        rng = np.random.default_rng(rng_seed)

        # ---- 0. Flatten the discrete grid into a list of vectors ----
        keys = list(self.config.param_grid.keys())
        value_lists = [self.config.param_grid[k] for k in keys]

        all_param_dicts: List[Dict[str, Any]] = []
        all_vectors: List[List[float]] = []

        def build_combinations(
            idx: int,
            current_dict: Dict[str, Any],
            current_vec: List[float],
        ):
            """Recursively build all parameter combinations and their numeric vectors."""
            if idx == len(keys):
                all_param_dicts.append(current_dict.copy())
                all_vectors.append(current_vec.copy())
                return
            key = keys[idx]
            for v in value_lists[idx]:
                current_dict[key] = v
                current_vec.append(float(v))
                build_combinations(idx + 1, current_dict, current_vec)
                current_vec.pop()

        build_combinations(0, {}, [])

        all_vectors = np.array(all_vectors, dtype=float)
        n_total = len(all_param_dicts)

        if n_init > n_total:
            n_init = n_total

        # ---- 1. Initial random evaluations ----
        all_indices = np.arange(n_total)
        rng.shuffle(all_indices)
        tried_indices = list(all_indices[:n_init])
        remaining_indices = set(all_indices[n_init:])

        X_obs = []
        y_obs = []

        best_params: Dict[str, Any] | None = None
        best_score: float | None = None

        for idx in tried_indices:
            params = all_param_dicts[idx]
            score = self.objective_fn(params)
            self.history.append({"params": params, "score": score})

            X_obs.append(all_vectors[idx])
            y_obs.append(score)

            if best_score is None:
                best_score = score
                best_params = params
            else:
                if self.config.maximize:
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

        X_obs = np.array(X_obs, dtype=float)
        y_obs = np.array(y_obs, dtype=float)

        # ---- 2. BO loop ----
        for _ in range(n_iter):
            if not remaining_indices:
                break

            # Fit GP surrogate on observed points
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gp.fit(X_obs, y_obs)

            # Build candidate matrix
            cand_idx = np.array(sorted(list(remaining_indices)))
            X_cand = all_vectors[cand_idx]

            mu, sigma = gp.predict(X_cand, return_std=True)

            # Acquisition: LCB/UCB style, but we always choose argmin
            if self.config.maximize:
                # maximize -> we want large (mu + kappa*sigma) -> minimize negative
                acq = -(mu + kappa * sigma)
            else:
                # minimize -> we want small (mu - kappa*sigma)
                acq = mu - kappa * sigma

            best_cand_pos = int(np.argmin(acq))
            chosen_global_idx = int(cand_idx[best_cand_pos])

            remaining_indices.remove(chosen_global_idx)
            tried_indices.append(chosen_global_idx)

            params = all_param_dicts[chosen_global_idx]
            score = self.objective_fn(params)
            self.history.append({"params": params, "score": score})

            # Update training data for the surrogate
            X_obs = np.vstack([X_obs, all_vectors[chosen_global_idx]])
            y_obs = np.append(y_obs, score)

            # Update best
            if self.config.maximize:
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                if score < best_score:
                    best_score = score
                    best_params = params

        return best_params, best_score, self.history

    # ----------------------------------------------------------------------
    # GENETIC ALGORITHM
    # ----------------------------------------------------------------------
    def genetic_search(
        self,
        population_size: int = 20,
        n_generations: int = 30,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elitism: int = 1,
        rng_seed: int = 0,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Genetic Algorithm over the discrete param_grid.

        Each individual is a full dict of hyperparameters.

        High-level steps:
        - Initialize a population of random configurations
        - For each generation:
            * keep the top `elitism` individuals unchanged
            * repeatedly:
                - select parents via tournament selection
                - apply one-point crossover
                - mutate offspring
        - At the end, return the best individual seen in the last generation.

        Parameters
        ----------
        population_size :
            Number of individuals per generation.
        n_generations :
            How many generations to evolve.
        tournament_size :
            Number of individuals participating in each tournament.
        crossover_rate :
            Probability of applying crossover to a pair of parents.
        mutation_rate :
            Probability of mutating an offspring.
        elitism :
            Number of top individuals that are copied directly to the next
            generation without change.
        rng_seed :
            Seed for reproducibility.

        Returns
        -------
        best_params : dict
        best_score : float
        history : list(dict)
        """
        import numpy as np

        rng = np.random.default_rng(rng_seed)

        keys = list(self.config.param_grid.keys())
        value_lists = [self.config.param_grid[k] for k in keys]

        # Make sure elitism is a valid number
        if elitism < 0:
            elitism = 0
        if elitism > population_size:
            elitism = population_size

        def random_individual() -> Dict[str, Any]:
            """Sample one random configuration from param_grid."""
            return {
                k: rng.choice(value_lists[i])
                for i, k in enumerate(keys)
            }

        def evaluate(indiv: Dict[str, Any]) -> float:
            """Evaluate one individual using the objective_fn and log it in history."""
            score = self.objective_fn(indiv)
            self.history.append({"params": indiv, "score": score})
            return float(score)

        def tournament_select(
            pop: List[Dict[str, Any]],
            fit: np.ndarray,
            k: int = tournament_size,
        ) -> Dict[str, Any]:
            """Pick one parent using tournament selection."""
            idxs = rng.integers(0, len(pop), size=k)
            if self.config.maximize:
                best_idx = idxs[np.argmax(fit[idxs])]
            else:
                best_idx = idxs[np.argmin(fit[idxs])]
            return pop[best_idx]

        def crossover(
            parent1: Dict[str, Any],
            parent2: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            One-point crossover over the ordered list of keys.

            We simply split the list of hyperparameters into two segments
            and swap the "tails" between parents.
            """
            if rng.random() > crossover_rate:
                return parent1.copy(), parent2.copy()

            # One-point crossover in the param order
            point = rng.integers(1, len(keys))
            child1: Dict[str, Any] = {}
            child2: Dict[str, Any] = {}
            for i, k in enumerate(keys):
                if i < point:
                    child1[k] = parent1[k]
                    child2[k] = parent2[k]
                else:
                    child1[k] = parent2[k]
                    child2[k] = parent1[k]
            return child1, child2

        def mutate(indiv: Dict[str, Any]) -> Dict[str, Any]:
            """
            Mutation: with some probability, change one random hyperparameter
            to another value from its candidate list.
            """
            if rng.random() > mutation_rate:
                return indiv
            # Change one random hyperparameter to another value
            i = rng.integers(0, len(keys))
            k = keys[i]
            indiv[k] = rng.choice(value_lists[i])
            return indiv

        # ---- 1. Initialize population ----
        population: List[Dict[str, Any]] = [
            random_individual() for _ in range(population_size)
        ]
        fitness = np.array([evaluate(ind) for ind in population], dtype=float)

        # ---- 2. Evolution loop ----
        for _ in range(n_generations):
            new_population: List[Dict[str, Any]] = []

            # --- Elitism: copy top individuals directly ---
            if elitism > 0:
                if self.config.maximize:
                    elite_indices = np.argsort(-fitness)[:elitism]
                else:
                    elite_indices = np.argsort(fitness)[:elitism]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())

            # --- Fill the rest of the population with offspring ---
            while len(new_population) < population_size:
                parent1 = tournament_select(population, fitness)
                parent2 = tournament_select(population, fitness)

                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)

                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

            population = new_population
            fitness = np.array([evaluate(ind) for ind in population], dtype=float)

        # ---- 3. Pick best individual in the final generation ----
        if self.config.maximize:
            best_idx = int(np.argmax(fitness))
        else:
            best_idx = int(np.argmin(fitness))

        best_params = population[best_idx]
        best_score = float(fitness[best_idx])

        return best_params, best_score, self.history
