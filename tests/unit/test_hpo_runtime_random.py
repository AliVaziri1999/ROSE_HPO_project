import pytest

from rose.hpo import runtime_random


def get_random_runtime_class():
    for name in ["RandomSearchRuntime", "RandomRuntime", "HpoRandomRuntime"]:
        cls = getattr(runtime_random, name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        "Could not find a random runtime class in runtime_random.py. "
        "Tried: RandomSearchRuntime, RandomRuntime, HpoRandomRuntime."
    )


def make_random_search_space():
    return {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "optimizer": {"type": "categorical", "choices": ["sgd", "adam"]},
    }


def test_random_configs_within_bounds():
    RandomRuntime = get_random_runtime_class()
    space = make_random_search_space()
    runtime = RandomRuntime(search_space=space, max_trials=100)

    for _ in range(50):
        cfg = runtime.propose()
        assert 1e-4 <= cfg["learning_rate"] <= 1e-2
        assert isinstance(cfg["num_layers"], int)
        assert 1 <= cfg["num_layers"] <= 3
        assert cfg["optimizer"] in ["sgd", "adam"]


def test_random_generates_correct_number_of_trials():
    RandomRuntime = get_random_runtime_class()
    space = make_random_search_space()
    runtime = RandomRuntime(search_space=space, max_trials=5)

    configs = []
    for _ in range(10):  # more than max_trials, ensure propose() stops
        cfg = runtime.propose()
        configs.append(cfg)

    # After max_trials, runtime should return None
    assert configs.count(None) >= 1
    num_real = sum(c is not None for c in configs)
    assert num_real == 5
