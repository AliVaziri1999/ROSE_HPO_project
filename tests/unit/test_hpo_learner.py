import numpy as np

from rose.hpo.hpo_learner import HPOLearner, HPOLearnerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quadratic_objective_min(params):
    """
    Simple convex objective: (x - 1)^2 + (y - 2)^2
    We want to MINIMIZE this.
    """
    x = float(params["x"])
    y = float(params["y"])
    return (x - 1.0) ** 2 + (y - 2.0) ** 2


def quadratic_objective_max(params):
    """
    Same function as above, but we pretend "bigger is better" by negating it.
    Used to test maximize=True path.
    """
    return -quadratic_objective_min(params)


def simple_param_grid():
    """
    Small 2D grid:
      x in {0, 1, 2}
      y in {1, 2}
    Global minimum of (x-1)^2 + (y-2)^2 is at (x=1, y=2).
    """
    return {
        "x": [0, 1, 2],
        "y": [1, 2],
    }


# ---------------------------------------------------------------------------
# GRID SEARCH TESTS
# ---------------------------------------------------------------------------

def test_grid_search_minimizes_over_full_cartesian_product():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=False)
    learner = HPOLearner(objective_fn=quadratic_objective_min, config=config)

    best_params, best_score, history = learner.grid_search()

    # Total combinations = 3 * 2 = 6
    assert len(history) == 6

    # Check that best is at (1, 2)
    assert best_params["x"] == 1
    assert best_params["y"] == 2

    # Best score should be 0 at (1, 2)
    assert best_score == 0.0


def test_grid_search_respects_maximize_flag():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=True)
    learner = HPOLearner(objective_fn=quadratic_objective_max, config=config)

    best_params, best_score, history = learner.grid_search()

    # Still expect (1, 2) because quadratic_objective_max is just -f
    assert best_params["x"] == 1
    assert best_params["y"] == 2

    # Best score should be 0 as well (negated minimum)
    assert best_score == 0.0

    # History must not be empty
    assert len(history) == 6


# ---------------------------------------------------------------------------
# RANDOM SEARCH TESTS
# ---------------------------------------------------------------------------

def test_random_search_samples_within_param_grid():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=False)
    learner = HPOLearner(objective_fn=quadratic_objective_min, config=config)

    n_samples = 10
    best_params, best_score, history = learner.random_search(
        n_samples=n_samples,
        rng_seed=42,
    )

    # Must evaluate exactly n_samples configurations
    assert len(history) == n_samples

    # All sampled params must lie inside the given candidate sets
    for entry in history:
        params = entry["params"]
        assert params["x"] in grid["x"]
        assert params["y"] in grid["y"]

    # Best_score must be consistent with history
    scores = [h["score"] for h in history]
    assert np.isclose(best_score, min(scores))


def test_random_search_is_reproducible_with_seed():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=False)

    # First run
    learner1 = HPOLearner(objective_fn=quadratic_objective_min, config=config)
    _, _, history1 = learner1.random_search(n_samples=5, rng_seed=123)

    # Second run with the same seed
    learner2 = HPOLearner(objective_fn=quadratic_objective_min, config=config)
    _, _, history2 = learner2.random_search(n_samples=5, rng_seed=123)

    # The sampled parameter sequences should match
    params_seq1 = [h["params"] for h in history1]
    params_seq2 = [h["params"] for h in history2]
    assert params_seq1 == params_seq2


# ---------------------------------------------------------------------------
# BAYESIAN SEARCH TESTS
# ---------------------------------------------------------------------------

def test_bayesian_search_produces_non_empty_history_and_valid_params():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=False)
    learner = HPOLearner(objective_fn=quadratic_objective_min, config=config)

    best_params, best_score, history = learner.bayesian_search(
        n_init=2,
        n_iter=3,
        kappa=1.0,
        rng_seed=0,
    )

    # History must not be empty
    assert len(history) > 0

    # All params in history must be from the discrete grid
    for entry in history:
        params = entry["params"]
        assert params["x"] in grid["x"]
        assert params["y"] in grid["y"]

    # Best score must be consistent with min(history scores)
    scores = [h["score"] for h in history]
    assert np.isclose(best_score, min(scores))

    # Best params must also lie in the grid
    assert best_params["x"] in grid["x"]
    assert best_params["y"] in grid["y"]


def test_bayesian_search_respects_maximize_flag():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=True)
    learner = HPOLearner(objective_fn=quadratic_objective_max, config=config)

    best_params, best_score, history = learner.bayesian_search(
        n_init=2,
        n_iter=3,
        kappa=1.0,
        rng_seed=0,
    )

    assert len(history) > 0

    scores = [h["score"] for h in history]
    # Now best_score should be the MAX over scores
    assert np.isclose(best_score, max(scores))


# ---------------------------------------------------------------------------
# GENETIC ALGORITHM TESTS
# ---------------------------------------------------------------------------

def test_genetic_search_keeps_params_inside_grid_and_tracks_history():
    grid = simple_param_grid()
    config = HPOLearnerConfig(param_grid=grid, maximize=False)
    learner = HPOLearner(objective_fn=quadratic_objective_min, config=config)

    population_size = 4
    n_generations = 3

    best_params, best_score, history = learner.genetic_search(
        population_size=population_size,
        n_generations=n_generations,
        tournament_size=2,
        crossover_rate=0.9,
        mutation_rate=0.5,
        elitism=1,
        rng_seed=0,
    )

    # Total evaluations = initial population + one eval per individual per generation
    expected_evals = population_size * (1 + n_generations)
    assert len(history) == expected_evals

    # All params must come from the discrete grid
    for entry in history:
        params = entry["params"]
        assert params["x"] in grid["x"]
        assert params["y"] in grid["y"]

    # Best params must also be valid
    assert best_params["x"] in grid["x"]
    assert best_params["y"] in grid["y"]

    # best_score should be a float
    assert isinstance(best_score, float)
