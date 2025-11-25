import pytest

from rose.hpo import hpo_learner


def get_learner_class():
    for name in ["HpoLearner", "HPOLearner", "HyperparameterLearner"]:
        cls = getattr(hpo_learner, name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        "Could not find learner class in hpo_learner.py. "
        "Tried: HpoLearner, HPOLearner, HyperparameterLearner."
    )


class DummyRuntime:
    def __init__(self, max_trials=3):
        self.max_trials = max_trials
        self.trials = 0

    def propose(self):
        if self.trials >= self.max_trials:
            return None
        self.trials += 1
        return {"x": self.trials}

    def observe(self, config, metric):
        # We just store metrics to show it was called
        self.last_observed = (config, metric)


def dummy_objective(config):
    # Very simple deterministic metric
    return -abs(config["x"] - 2)


def test_learner_stops_after_max_trials():
    Learner = get_learner_class()
    runtime = DummyRuntime(max_trials=4)
    learner = Learner(runtime=runtime, objective=dummy_objective)

    result = learner.run()  # adapt if method name different

    # DummyRuntime will produce exactly 4 proposals
    assert runtime.trials == 4

    # The best metric is at x=2 with value 0
    best_cfg, best_metric = result
    assert best_cfg["x"] == 2
    assert best_metric == pytest.approx(0.0)
