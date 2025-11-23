import asyncio
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger

from rose.hpo import HPOLearner, HPOLearnerConfig
from rose.hpo.runtime import run_grid_search_distributed


def build_sine_dataset(
    n_train: int = 40,
    n_test: int = 100,
    noise: float = 0.1,
    seed: int = 42,
):
    """
    Simple toy dataset:
        y = sin(x) + Gaussian noise
    """
    rng = np.random.default_rng(seed)

    X_train = np.linspace(0, 4 * np.pi, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + rng.normal(0, noise, size=n_train)

    X_test = np.linspace(0, 4 * np.pi, n_test).reshape(-1, 1)
    y_test = np.sin(X_test).ravel()

    return X_train, y_train, X_test, y_test


def make_objective(X_train, y_train, X_test, y_test):
    """
    Wrap GP training + evaluation into an objective(params) -> RMSE.
    """

    def objective(params: dict) -> float:
        length_scale = float(params["length_scale"])
        noise_level = float(params["noise_level"])

        kernel = RBF(length_scale=length_scale) + WhiteKernel(
            noise_level=noise_level
        )
        model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse  # we minimize this

    return objective


async def main():
    # 1) Configure ROSE / asyncflow runtime
    init_default_logger()

    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend)

    # 2) Prepare synthetic data + objective
    X_train, y_train, X_test, y_test = build_sine_dataset()
    objective = make_objective(X_train, y_train, X_test, y_test)

    # 3) Define hyperparameter grid
    search_space = {
        "length_scale": [0.2, 0.5, 1.0, 2.0],
        "noise_level": [0.01, 0.05, 0.1],
    }

    config = HPOLearnerConfig(
        param_grid=search_space,
        maximize=False,  # we minimize RMSE
    )

    # 4) Create HPOLearner and run **distributed** grid search
    hpo = HPOLearner(objective_fn=objective, config=config)

    result = await run_grid_search_distributed(asyncflow, hpo)

    best_params = result["best_params"]
    best_score = result["best_score"]
    history = result["history"]

    print("\n=== HPO via ROSE runtime (GRID, distributed) ===")
    print(f"Best params: {best_params}")
    print(f"Best score (RMSE): {best_score:.4f}")
    print(f"Total configs tried: {len(history)}")

    print("\n=== Result object returned to main() ===")
    print(result)

    # 5) Shutdown runtime cleanly
    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
