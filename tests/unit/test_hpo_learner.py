import math

from rose.hpo import HPOLearner, HPOLearnerConfig


def make_quadratic_objective(center: float = 2.0):
    """Simple 1D objective: (x - center)^2, minimum at x = center."""
    def objective(params: dict) -> float:
        x = float(params["x"])
        return (x - center) ** 2
    return objective


def test_grid_search_finds_exact_minimum():
    # x in {0, 1, 2, 3, 4} – minimum at x = 2
    config = HPOLearnerConfig(
        param_grid={"x": [0, 1, 2, 3, 4]},
        maximize=False,
    )
    objective = make_quadratic_objective(center=2.0)
    learner = HPOLearner(objective_fn=objective, config=config)

    best_params, best_score, history = learner.grid_search()

    assert best_params["x"] == 2
    assert best_score == 0.0
    # should have evaluated all 5 configurations
    assert len(history) == 5
    # every history entry has the right structure
    for item in history:
        assert "params" in item and "score" in item
        assert "x" in item["params"]


def test_random_search_respects_param_grid():
    config = HPOLearnerConfig(
        param_grid={"x": [-1, 0, 1]},
        maximize=False,
    )
    objective = make_quadratic_objective(center=0.0)
    learner = HPOLearner(objective_fn=objective, config=config)

    n_samples = 10
    best_params, best_score, history = learner.random_search(
        n_samples=n_samples,
        rng_seed=123,
    )

    # correct history length
    assert len(history) == n_samples

    # all sampled x values must come from the grid
    allowed = set(config.param_grid["x"])
    for item in history:
        assert item["params"]["x"] in allowed
        assert isinstance(item["score"], (float, int))

    # minimum achievable value is 0 (at x = 0)
    assert math.isclose(best_score, 0.0, rel_tol=1e-10)
    assert best_params["x"] == 0


def test_bayesian_search_returns_valid_result():
    # small 2D grid so the test is fast
    config = HPOLearnerConfig(
        param_grid={
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0],
        },
        maximize=False,
    )

    def objective(params: dict) -> float:
        x = float(params["x"])
        y = float(params["y"])
        # simple bowl-shaped function, min at (1, 0)
        return (x - 1.0) ** 2 + (y - 0.0) ** 2

    learner = HPOLearner(objective_fn=objective, config=config)

    best_params, best_score, history = learner.bayesian_search(
        n_init=3,
        n_iter=5,
        kappa=2.0,
        rng_seed=0,
    )

    # history should not be empty
    assert len(history) >= 3

    # best params must lie inside the discrete grid
    assert best_params["x"] in config.param_grid["x"]
    assert best_params["y"] in config.param_grid["y"]

    # global minimum value is 0 at (1, 0), allow a tiny numerical tolerance
    assert best_score >= -1e-12
    assert best_score <= 1e-6
