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

    This first version is **pure Python**:
    - no async, no WorkflowEngine, no HPC
    - just loops over all combinations in `param_grid`
      and calls a user-provided objective function.

    Later we will connect this to the full ROSE runtime.
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
                    return rmse

        config :
            HPOLearnerConfig with param_grid and maximize flag.
        """
        self.objective_fn = objective_fn
        self.config = config

        # store all tried configs + scores
        self.history: List[Dict[str, Any]] = []

    def _iter_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations from param_grid."""
        keys = list(self.config.param_grid.keys())
        value_lists = [self.config.param_grid[k] for k in keys]

        for combo in itertools.product(*value_lists):
            params = {k: v for k, v in zip(keys, combo)}
            yield params

    def grid_search(self) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Run a basic grid search.

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

        # defensive, should not happen if grid is non–empty
        if best_params is None or best_score is None:
            raise ValueError("param_grid is empty – nothing to search over.")

        return best_params, best_score, self.history

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

        best_params = None
        best_score = None

        for _ in range(n_samples):

            # choose random value for each hyperparameter
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

        def build_combinations(idx: int, current_dict: Dict[str, Any], current_vec: List[float]):
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

        best_params = None
        best_score = None

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

            # update datasets
            X_obs = np.vstack([X_obs, all_vectors[chosen_global_idx]])
            y_obs = np.append(y_obs, score)

            # update best
            if self.config.maximize:
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                if score < best_score:
                    best_score = score
                    best_params = params

        return best_params, best_score, self.history


    