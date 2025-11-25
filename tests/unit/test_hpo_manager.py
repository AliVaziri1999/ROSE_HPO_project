import pytest

from rose.hpo import manager, runtime_grid, runtime_random, runtime_bayesian


def get_manager_class():
    for name in ["HpoManager", "HPOManager", "HyperparameterManager"]:
        cls = getattr(manager, name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        "Could not find manager class in manager.py. "
        "Tried: HpoManager, HPOManager, HyperparameterManager."
    )


def test_manager_selects_correct_runtime_by_name():
    HpoManager = get_manager_class()

    mgr_grid = HpoManager(strategy="grid")
    assert isinstance(
        mgr_grid.runtime,
        (getattr(runtime_grid, "GridSearchRuntime", object),
         getattr(runtime_grid, "GridRuntime", object)),
    )

    mgr_random = HpoManager(strategy="random")
    assert isinstance(
        mgr_random.runtime,
        (getattr(runtime_random, "RandomSearchRuntime", object),
         getattr(runtime_random, "RandomRuntime", object)),
    )

    mgr_bayes = HpoManager(strategy="bayesian")
    assert isinstance(
        mgr_bayes.runtime,
        (getattr(runtime_bayesian, "BayesianSearchRuntime", object),
         getattr(runtime_bayesian, "BayesianRuntime", object)),
    )


def test_manager_tracks_best_config_and_score():
    HpoManager = get_manager_class()
    mgr = HpoManager(strategy="grid")

    # Assume manager has an API like: record_result(config, metric)
    cfg1 = {"lr": 0.001}
    cfg2 = {"lr": 0.01}

    mgr.record_result(cfg1, metric=0.7)
    mgr.record_result(cfg2, metric=0.9)

    assert mgr.best_config == cfg2
    assert mgr.best_score == pytest.approx(0.9)
