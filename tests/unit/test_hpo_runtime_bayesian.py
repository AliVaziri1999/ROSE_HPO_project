import pytest

from rose.hpo import runtime_bayesian


def get_bayesian_runtime_class():
    for name in ["BayesianSearchRuntime", "BayesianRuntime", "HpoBayesianRuntime"]:
        cls = getattr(runtime_bayesian, name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        "Could not find a bayesian runtime class in runtime_bayesian.py. "
        "Tried: BayesianSearchRuntime, BayesianRuntime, HpoBayesianRuntime."
    )


def make_bayesian_search_space():
    return {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2},
        "num_layers": {"type": "int", "low": 1, "high": 3},
    }


def test_bayesian_initial_design():
    BayesianRuntime = get_bayesian_runtime_class()
    space = make_bayesian_search_space()

    # Adjust keyword args if your class expects different ones
    runtime = BayesianRuntime(search_space=space, initial_points=3, max_trials=5)

    configs = []
    for _ in range(3):
        cfg = runtime.propose()
        configs.append(cfg)

    assert len(configs) == 3
    for cfg in configs:
        assert 1e-4 <= cfg["learning_rate"] <= 1e-2
        assert 1 <= cfg["num_layers"] <= 3


def test_bayesian_updates_after_observations():
    BayesianRuntime = get_bayesian_runtime_class()
    space = make_bayesian_search_space()
    runtime = BayesianRuntime(search_space=space, initial_points=2, max_trials=4)

    # First two points
    cfg1 = runtime.propose()
    cfg2 = runtime.propose()

    # Fake metrics (e.g., accuracy)
    runtime.observe(cfg1, metric=0.8)
    runtime.observe(cfg2, metric=0.9)

    # Next propose() should work and use the updated surrogate internally
    cfg3 = runtime.propose()
    assert cfg3 is not None
    assert 1e-4 <= cfg3["learning_rate"] <= 1e-2
    assert 1 <= cfg3["num_layers"] <= 3
