from __future__ import annotations

import asyncio
from typing import Any, Dict, List
import numpy as np

from radical.asyncflow import WorkflowEngine

from .hpo_learner import HPOLearner


# Small helper for JSON serialization of NumPy types
def _json_default(obj):
    """
    Convert NumPy types to plain Python types so json.dumps can handle them.
    """
    import numpy as _np

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


# ===========================================================================
# INTERNAL HELPER — distributed evaluation of multiple configs
# Added:
#   - return_exceptions=True for fault tolerance
#   - checkpointing argument (path + frequency)
# ===========================================================================
async def _evaluate_configs_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    param_list: List[Dict[str, Any]],
    context_label: str = "HPO",
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate each params dict as a separate ROSE task using asyncflow.

    Improvements included:
      - fault-tolerant gather() (per-task exception handling)
      - checkpointing
      - shared by Grid, Random, Bayesian, GA distributed versions.

    Parameters
    ----------
    checkpoint_path : str or None
        If not None, write the full hpo.history as JSON to this path
        periodically during the run.
    checkpoint_freq : int
        Save a checkpoint every `checkpoint_freq` successful evaluations.
        If <= 0, checkpointing is disabled.
    """

    if not param_list:
        raise ValueError(
            f"{context_label} received an empty param_list – nothing to evaluate."
        )

    print(f"[{context_label}] Submitting {len(param_list)} configs to ROSE...")

    local_history: List[Dict[str, Any]] = []

    # Worker function (runs inside ROSE workers)
    async def eval_one(params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            score = hpo.objective_fn(params)
            return {"params": params, "score": float(score)}
        except Exception as e:
            # We capture the error instead of crashing the whole run
            return {"params": params, "error": str(e), "score": None}

    eval_task = asyncflow.function_task(eval_one)
    futures = [eval_task(p) for p in param_list]

    # Fault-tolerant gather: we also handle AsyncFlow-level exceptions
    results = await asyncio.gather(*futures, return_exceptions=True)

    best_params = None
    best_score: float | None = None

    for params, rec in zip(param_list, results):

        # If the task itself crashed inside AsyncFlow
        if isinstance(rec, Exception):
            print(f"[{context_label}] ERROR (AsyncFlow-level) for {params}: {rec}")
            hpo.history.append({"params": params, "score": None, "error": str(rec)})
            continue

        # If our eval returned an error (inside objective_fn)
        if "error" in rec:
            print(f"[{context_label}] ERROR (objective_fn) for {params}: {rec['error']}")
            hpo.history.append(rec)
            continue

        # Normal success path
        score = rec["score"]
        hpo.history.append(rec)
        local_history.append(rec)

        # Track best
        if best_score is None:
            best_score = score
            best_params = rec["params"]
        else:
            if hpo.config.maximize and score > best_score:
                best_score = score
                best_params = rec["params"]
            elif not hpo.config.maximize and score < best_score:
                best_score = score
                best_params = rec["params"]

        # checkpointing every N successful evaluations
        if (
            checkpoint_path is not None
            and checkpoint_freq > 0
            and len(local_history) > 0
            and len(local_history) % checkpoint_freq == 0
        ):
            import json
            import pathlib
            import os

            # Project root: ROSE_HPO_project/
            project_root = pathlib.Path(__file__).resolve().parents[2]
            default_dir = project_root / "checkpoints"

            raw_path = pathlib.Path(checkpoint_path)

            # If user passed a relative name, always place under ./checkpoints/
            if not raw_path.is_absolute():
                path = default_dir / raw_path
            else:
                path = raw_path

            path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(
                json.dumps(hpo.history, default=_json_default, indent=2)
            )
            print(f"[{context_label}] Checkpoint saved → {path}")

    if best_params is None or best_score is None:
        raise ValueError(f"[{context_label}] No valid results – all configs failed.")

    print(
        f"[{context_label}] Best score: {best_score:.4f} "
        f"with params: {best_params}"
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": hpo.history,
    }


# ===========================================================================
#                       DISTRIBUTED GRID SEARCH
# ===========================================================================
async def run_grid_search_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Distributed GRID search: one config → one ROSE task.
    """
    param_list = list(hpo._iter_param_combinations())
    return await _evaluate_configs_distributed(
        asyncflow=asyncflow,
        hpo=hpo,
        param_list=param_list,
        context_label="GRID",
        checkpoint_path=checkpoint_path,
        checkpoint_freq=checkpoint_freq,
    )


# ===========================================================================
#                       DISTRIBUTED RANDOM SEARCH
# ===========================================================================
async def run_random_search_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    n_samples: int,
    rng_seed: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Distributed RANDOM search: sample n configs from the grid and evaluate them.
    """
    param_grid = hpo.config.param_grid
    if not param_grid:
        raise ValueError("param_grid is empty – nothing to sample from.")

    rng = np.random.default_rng(rng_seed)
    keys = list(param_grid.keys())
    arrays = [np.array(param_grid[k]) for k in keys]

    sampled = [
        {k: arrays[i][rng.integers(0, len(arrays[i]))] for i, k in enumerate(keys)}
        for _ in range(n_samples)
    ]

    return await _evaluate_configs_distributed(
        asyncflow=asyncflow,
        hpo=hpo,
        param_list=sampled,
        context_label="RANDOM",
        checkpoint_path=checkpoint_path,
        checkpoint_freq=checkpoint_freq,
    )


# ===========================================================================
#                       DISTRIBUTED BAYESIAN OPTIMIZATION
# ===========================================================================
async def run_bayesian_search_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    n_init: int = 5,
    n_iter: int = 15,
    kappa: float = 2.0,
    rng_seed: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Distributed Bayesian Optimization.
    Mirrors HPOLearner.bayesian_search(), but each evaluation is done
    through ROSE tasks.
    """

    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(rng_seed)

    # Convert param_grid → arrays
    param_list_full = list(hpo._iter_param_combinations())
    keys = list(hpo.config.param_grid.keys())
    vectors = np.array([[float(p[k]) for k in keys] for p in param_list_full])

    n_total = len(param_list_full)
    if n_total == 0:
        raise ValueError("param_grid is empty.")

    if n_init > n_total:
        n_init = n_total

    idxs = np.arange(n_total)
    rng.shuffle(idxs)

    tried = list(idxs[:n_init])
    remaining = set(idxs[n_init:])

    # ---- Initial evaluations (parallel) ----
    init_params = [param_list_full[i] for i in tried]
    start = len(hpo.history)

    await _evaluate_configs_distributed(
        asyncflow,
        hpo,
        init_params,
        context_label="BAYESIAN_INIT",
        checkpoint_path=checkpoint_path,
        checkpoint_freq=checkpoint_freq,
    )

    batch = hpo.history[start:]
    X_obs = np.array([[float(rec["params"][k]) for k in keys] for rec in batch])
    y_obs = np.array([rec["score"] for rec in batch])

    best_params = None
    best_score = None
    for rec in batch:
        if best_score is None:
            best_score = rec["score"]
            best_params = rec["params"]
        else:
            if hpo.config.maximize and rec["score"] > best_score:
                best_params = rec["params"]
                best_score = rec["score"]
            elif not hpo.config.maximize and rec["score"] < best_score:
                best_params = rec["params"]
                best_score = rec["score"]

    # ---- BO iterations ----
    for _ in range(n_iter):
        if not remaining:
            break

        gp = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(1e-3),
            normalize_y=True,
        )
        gp.fit(X_obs, y_obs)

        cand_idx = np.array(sorted(list(remaining)))
        X_cand = vectors[cand_idx]
        mu, sigma = gp.predict(X_cand, return_std=True)

        # UCB/LCB acquisition
        if hpo.config.maximize:
            acq = -(mu + kappa * sigma)
        else:
            acq = (mu - kappa * sigma)

        next_idx = cand_idx[int(np.argmin(acq))]
        remaining.remove(next_idx)

        p = param_list_full[next_idx]

        start = len(hpo.history)
        await _evaluate_configs_distributed(
            asyncflow,
            hpo,
            [p],
            context_label="BAYESIAN",
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
        )
        rec = hpo.history[start]

        x_vec = np.array([float(rec["params"][k]) for k in keys])
        score = rec["score"]

        # update sets
        X_obs = np.vstack([X_obs, x_vec])
        y_obs = np.append(y_obs, score)

        # update best
        if hpo.config.maximize:
            if score > best_score:
                best_score = score
                best_params = rec["params"]
        else:
            if score < best_score:
                best_score = score
                best_params = rec["params"]

    print(f"[BAYESIAN] Distributed BO → best score {best_score} params {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": hpo.history,
    }


# ===========================================================================
#                       DISTRIBUTED GENETIC ALGORITHM
# ===========================================================================
async def run_genetic_search_distributed(
    asyncflow: WorkflowEngine,
    hpo: HPOLearner,
    population_size: int = 20,
    n_generations: int = 30,
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    elitism: int = 1,
    rng_seed: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_freq: int = 10,
) -> Dict[str, Any]:
    """
    Distributed evaluation version of HPOLearner.genetic_search().
    """

    import numpy as np

    rng = np.random.default_rng(rng_seed)

    keys = list(hpo.config.param_grid.keys())
    value_lists = [hpo.config.param_grid[k] for k in keys]

    def random_individual():
        return {
            k: rng.choice(value_lists[i])
            for i, k in enumerate(keys)
        }

    def crossover(p1, p2):
        if rng.random() > crossover_rate:
            return p1.copy(), p2.copy()
        point = rng.integers(1, len(keys))
        c1, c2 = {}, {}
        for i, k in enumerate(keys):
            if i < point:
                c1[k] = p1[k]
                c2[k] = p2[k]
            else:
                c1[k] = p2[k]
                c2[k] = p1[k]
        return c1, c2

    def mutate(ind):
        if rng.random() > mutation_rate:
            return ind
        i = rng.integers(0, len(keys))
        k = keys[i]
        ind[k] = rng.choice(value_lists[i])
        return ind

    # ----- initial population eval -----
    population = [random_individual() for _ in range(population_size)]
    start = len(hpo.history)

    await _evaluate_configs_distributed(
        asyncflow,
        hpo,
        population,
        context_label="GA_INIT",
        checkpoint_path=checkpoint_path,
        checkpoint_freq=checkpoint_freq,
    )

    batch = hpo.history[start:]
    fitness = np.array([rec["score"] for rec in batch])

    # ----- evolution loop -----
    for _ in range(n_generations):
        new_pop = []

        # elitism
        if elitism > 0 and len(fitness) > 0:
            if hpo.config.maximize:
                elite = np.argsort(-fitness)[:elitism]
            else:
                elite = np.argsort(fitness)[:elitism]
            for idx in elite:
                new_pop.append(population[idx].copy())

        # offspring creation
        while len(new_pop) < population_size and len(fitness) > 0:
            # tournament selection
            idxs = rng.integers(0, len(population), size=tournament_size)
            if hpo.config.maximize:
                p1 = population[idxs[np.argmax(fitness[idxs])]]
                p2 = population[idxs[np.argsort(fitness[idxs])[-2]]]
            else:
                p1 = population[idxs[np.argmin(fitness[idxs])]]
                p2 = population[idxs[np.argsort(fitness[idxs])[1]]]

            c1, c2 = crossover(p1, p2)
            c1, c2 = mutate(c1), mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        population = new_pop

        # distributed eval of next generation
        start = len(hpo.history)
        await _evaluate_configs_distributed(
            asyncflow,
            hpo,
            population,
            context_label="GA",
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
        )

        batch = hpo.history[start:]
        fitness = np.array([rec["score"] for rec in batch])

    # pick best
    if len(fitness) == 0:
        raise ValueError("[GA] No valid individuals evaluated – all scores are None.")

    if hpo.config.maximize:
        idx = int(np.argmax(fitness))
    else:
        idx = int(np.argmin(fitness))

    best_params = population[idx]
    best_score = float(fitness[idx])

    print(f"[GA] Distributed GA → best {best_score} params {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": hpo.history,
    }
