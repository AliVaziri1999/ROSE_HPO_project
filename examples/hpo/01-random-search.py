"""
Random search example for Gaussian Process Regression.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from rose.hpo import HPOLearner, HPOLearnerConfig


# 1. synthetic data
def make_data(seed=42):
    rng = np.random.default_rng(seed)
    X_train = np.linspace(0, 4 * np.pi, 40).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + rng.normal(0, 0.1, X_train.shape[0])

    X_test = np.linspace(0, 4 * np.pi, 80).reshape(-1, 1)
    y_test = np.sin(X_test).ravel()
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = make_data()


# 2. objective fn
def objective(params: dict) -> float:
    length_scale = params["length_scale"]
    noise_level = params["noise_level"]

    kernel = RBF(length_scale) + WhiteKernel(noise_level=noise_level)
    model = GaussianProcessRegressor(kernel=kernel)

    model.fit(X_train, y_train)
    y_pred, _ = model.predict(X_test, return_std=True)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Random try: {params} → RMSE={rmse:.4f}")
    return rmse


# 3. hyperparameter space
param_grid = {
    "length_scale": [0.2, 0.5, 1.0, 2.0, 5.0],
    "noise_level": [0.005, 0.01, 0.05, 0.1],
}

config = HPOLearnerConfig(param_grid=param_grid, maximize=False)

# 4. run random search
learner = HPOLearner(objective, config)
best_params, best_score, history = learner.random_search(n_samples=20)

print("\n===== RANDOM SEARCH RESULT =====")
print(f"Tried {len(history)} configurations")
print("Best params:", best_params)
print(f"Best RMSE: {best_score:.4f}")
print("================================")
