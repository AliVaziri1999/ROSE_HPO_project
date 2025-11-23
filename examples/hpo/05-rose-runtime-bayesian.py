import asyncio
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: F401
from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # noqa: F401
from sklearn.metrics import mean_squared_error

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger

from rose.hpo import HPOLearner, HPOLearnerConfig


# ---------------------------------------------------------------------------
# 1. Data + objective function (same structure as your other examples)
# ---------------------------------------------------------------------------

def build_sine_dataset(
    n_train: int = 40,
    n_test: int = 100,
    noise: float = 0.1,
    seed: int = 42,
):
    """
    Simple 1D regression toy problem:

        y = sin(x) + Gaussian noise

    Returns:
        X_train, y_train, X_test, y_test
    """
    rng = np.random.default_rng(seed)

    X_train = np.linspace(0, 4 * np.pi, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + rng.normal(0, noise, size=n_train)

    X_test = np.linspace(0, 4 * np.pi, n_test).reshape(-1, 1)
    y_test = np.sin(X_test).ravel()

    return X_train, y_train, X_test, y_test


def make_objective(X_train, y_train, X_test, y_test):
    """
    Wrap GP training + evaluation in a function that HPOLearner can call
    with a hyperparameter dict.
    """

    def objective(params: dict) -> float:
        length_scale = float(params["length_scale"])
        noise_level = float(params["noise_level"])

        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse  # HPOLearner will *minimize* this

    return objective


# ---------------------------------------------------------------------------
# 2. Main: run Bayesian search via ROSE runtime
# ---------------------------------------------------------------------------

async def main():
    """
    Run Bayesian Optimization HPO through the ROSE / asyncflow runtime.

    The actual Bayesian search is implemented in HPOLearner.bayesian_search,
    and we just submit one HPO job as a ROSE task.
    """

    # 2.1 Set up logging + execution backend
    init_default_logger()
    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    # 2.2 Build dataset and objective
    X_train, y_train, X_test, y_test = build_sine_dataset()
    objective = make_objective(X_train, y_train, X_test, y_test)

    # 2.3 Define discrete hyperparameter grid
    search_space = {
        "length_scale": [0.2, 0.5, 1.0, 2.0],
        "noise_level": [0.01, 0.05, 0.1],
    }

    # 2.4 Configure the HPO learner (minimize RMSE → maximize=False)
    config = HPOLearnerConfig(
        param_grid=search_space,
        maximize=False,
    )

    # 2.5 Create the local (synchronous) HPO helper
    hpo = HPOLearner(
        objective_fn=objective,
        config=config,
    )

    # 2.6 Define a pure sync function that runs Bayesian search once
    def _do_bayes():
        """
        Synchronous Bayesian search.
        Runs entirely inside a worker process (via run_in_executor).
        """
        best_params, best_score, history = hpo.bayesian_search(
            n_init=5,
            n_iter=15,
            kappa=2.0,
            rng_seed=42,
        )

        print("\n===== BAYESIAN OPTIMIZATION (via ROSE runtime) =====")
        print(f"Total evaluations: {len(history)}")
        print(f"Best params: {best_params}")
        print(f"Best RMSE : {best_score:.4f}")
        print("====================================================\n")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "history": history,
        }

    # 2.7 Wrap it in an async function so asyncflow.function_task is happy
    async def run_hpo_once():
        """
        Async wrapper that offloads the CPU-bound Bayesian search
        to a background executor thread.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _do_bayes)
        return result

    # 2.8 Submit as a ROSE function task
    hpo_future = asyncflow.function_task(run_hpo_once)()
    best_result = await hpo_future

    print("=== Result object returned to main() ===")
    print(best_result)

    # 2.9 Clean shutdown of the runtime
    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

