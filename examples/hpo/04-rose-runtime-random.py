import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger


# ---------------------------------------------------------------------------
# 1) Data + objective
# ---------------------------------------------------------------------------

def build_sine_dataset(n_train: int = 40,
                       n_test: int = 100,
                       noise: float = 0.1,
                       seed: int = 42):
    """Same simple y = sin(x) + noise dataset as other examples."""
    rng = np.random.default_rng(seed)

    X_train = np.linspace(0, 4 * np.pi, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + rng.normal(0, noise, size=n_train)

    X_test = np.linspace(0, 4 * np.pi, n_test).reshape(-1, 1)
    y_test = np.sin(X_test).ravel()

    return X_train, y_train, X_test, y_test


def make_objective(X_train, y_train, X_test, y_test):
    """
    Wrap GP training/eval into a pure function that receives params and
    returns a scalar score (RMSE).
    """
    def objective(params: Dict[str, Any]) -> float:
        length_scale = float(params["length_scale"])
        noise_level = float(params["noise_level"])

        kernel = RBF(length_scale=length_scale) + WhiteKernel(
            noise_level=noise_level
        )
        model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return float(rmse)

    return objective


# ---------------------------------------------------------------------------
# 2) Helper: draw random configs from a discrete grid
# ---------------------------------------------------------------------------

def sample_random_configs(
    param_grid: Dict[str, List[Any]],
    n_samples: int,
    rng_seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Given a discrete param_grid = {name: [values...]}, sample n_samples
    random configurations (with replacement).
    """
    rng = np.random.default_rng(rng_seed)
    keys = list(param_grid.keys())
    value_arrays = [np.array(param_grid[k]) for k in keys]

    configs: List[Dict[str, Any]] = []

    for _ in range(n_samples):
        cfg = {
            k: value_arrays[i][rng.integers(0, len(value_arrays[i]))]
            for i, k in enumerate(keys)
        }
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# 3) Main: RANDOM search but each sampled config is its OWN ROSE task
# ---------------------------------------------------------------------------

async def main():
    # 3.1 init logging + backend + workflow engine
    init_default_logger()
    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    # 3.2 build dataset + objective
    X_train, y_train, X_test, y_test = build_sine_dataset()
    objective = make_objective(X_train, y_train, X_test, y_test)

    # 3.3 define discrete search space
    param_grid = {
        "length_scale": [0.2, 0.5, 1.0, 2.0, 5.0],
        "noise_level": [0.005, 0.01, 0.05, 0.1],
    }

    # how many random configs we want to try
    n_samples = 20
    random_configs = sample_random_configs(param_grid, n_samples, rng_seed=42)

    # 3.4 define a ROSE task for a SINGLE random configuration
    @asyncflow.function_task
    async def eval_one(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ROSE task body: evaluate GP on a single hyperparameter config.
        We wrap the CPU-bound part into run_in_executor so the async
        event loop is not blocked.
        """
        loop = asyncio.get_running_loop()

        def _do_eval() -> Dict[str, Any]:
            score = objective(params)
            return {"params": params, "score": float(score)}

        return await loop.run_in_executor(None, _do_eval)

    # 3.5 submit all random configs in PARALLEL
    tasks = [eval_one(cfg) for cfg in random_configs]
    results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

    # 3.6 aggregate best result
    best = min(results, key=lambda r: r["score"])
    best_params = best["params"]
    best_score = best["score"]

    print("\n===== RANDOM SEARCH (distributed, via ROSE runtime) =====")
    print(f"Tried {len(results)} random configurations")
    print(f"Best params: {best_params}")
    print(f"Best RMSE : {best_score:.4f}")
    print("=========================================================")

    print("\n=== Result object returned to main() ===")
    print(best)

    # 3.7 shut down the workflow
    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
