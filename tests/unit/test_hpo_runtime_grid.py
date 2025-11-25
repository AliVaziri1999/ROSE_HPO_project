import itertools
import math
import pytest

from rose.hpo import runtime_grid


def get_grid_runtime_class():
    # Adjust here if your class has a different name
    for name in ["GridSearchRuntime", "GridRuntime", "HpoGridRuntime"]:
        cls = getattr(runtime_grid, name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        "Could not find a grid runtime class in runtime_grid.py. "
        "Tried: GridSearchRuntime, GridRuntime, HpoGridRuntime."
    )


def make_simple_search_space():
    # Example search space: 2 * 2 = 4 combos
    return {
        "learning_rate": [0.001, 0.01],
        "num_layers": [1, 2],
    }


def test_grid_generates_full_cartesian_product():
    GridRuntime = get_grid_runtime_class()
    space = make_simple_search_space()

    runtime = GridRuntime(search_space=space)  # adjust kwargs if needed

    # Collect all proposed configs until exhaustion or a safety cap
    configs = []
    for _ in range(10):  # safety cap
        cfg = runtime.propose()
        if cfg is None:
            break
        configs.append(cfg)

    # expected cartesian product
    expected = list(
        itertools.product(space["learning_rate"], space["num_layers"])
    )
    assert len(configs) == len(expected)

    observed_tuples = {
        (cfg["learning_rate"], cfg["num_layers"]) for cfg in configs
    }
    expected_tuples = set(expected)

    assert observed_tuples == expected_tuples


def test_grid_does_not_repeat_configs():
    GridRuntime = get_grid_runtime_class()
    space = make_simple_search_space()
    runtime = GridRuntime(search_space=space)

    seen = set()
    while True:
        cfg = runtime.propose()
        if cfg is None:
            break
        key = (cfg["learning_rate"], cfg["num_layers"])
        assert key not in seen, "Grid runtime returned a duplicate config"
        seen.add(key)

    # Ensure we visited all configs
    assert len(seen) == len(space["learning_rate"]) * len(space["num_layers"])
