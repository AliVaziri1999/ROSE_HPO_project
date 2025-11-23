"""
Basic example: grid search over Gaussian Process hyperparameters.

This script:
  1. Creates a small synthetic regression dataset.
  2. Defines an objective() function that trains a GPR model and
     returns RMSE on a test set.
  3. Uses HPOLearner + HPOLearnerConfig to run a simple grid search.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from rose.hpo import HPOLearner, HPOLearnerConfig


# 1. Prepare synthetic data (very small & fast)
def make_data(random_seed: int = 42):
    rng = np.random.default_rng(random_seed)

    # Train data
    X_train = np.linspace(0, 4 * np.pi, 40).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + rng.normal(0, 0.1, X_train.shape[0])

    # Test data
    X_test = np.linspace(0, 4 * np.pi, 80).reshape(-1, 1)
    y_test = np.sin(X_test).ravel()

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = make_data()


# 2. Define the objective function for HPO
def objective(params: dict) -> float:
    """
    Train a GaussianProcessRegressor with given hyperparameters
    and return the RMSE on the test set.

    params dict contains:
      - "length_scale"
      - "noise_level"
    """
    length_scale = params["length_scale"]
    noise_level = params["noise_level"]

    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

    model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model.fit(X_train, y_train)

    y_pred, _ = model.predict(X_test, return_std=True)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(
        f" Tried params: length_scale={length_scale:.3f}, "
        f"noise_level={noise_level:.3f} -> RMSE={rmse:.4f}"
    )

    # We want to **minimize** RMSE, so we return it directly.
    return rmse


# 3. Define the grid we want to search
param_grid = {
    "length_scale": [0.2, 0.5, 1.0, 2.0],
    "noise_level": [0.01, 0.05, 0.1],
}

config = HPOLearnerConfig(param_grid=param_grid, maximize=False)

# 4. Create learner and run grid search
learner = HPOLearner(objective_fn=objective, config=config)

best_params, best_score, history = learner.grid_search()

print("\n================ GRID SEARCH SUMMARY ================")
print(f"Number of configurations tried: {len(history)}")
print(f"Best params: {best_params}")
print(f"Best RMSE : {best_score:.4f}")
print("====================================================")

