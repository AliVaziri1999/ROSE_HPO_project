import asyncio

import pytest

radical_asyncflow = pytest.importorskip("radical.asyncflow")
from radical.asyncflow import WorkflowEngine, RadicalExecutionBackend

from rose.hpo import hpo_learner, runtime_grid


@pytest.mark.asyncio
async def test_hpo_runs_with_local_asyncflow():
    # Minimal execution backend on local.localhost
    execution_engine = await RadicalExecutionBackend(
        {"runtime": 10, "resource": "local.localhost"}
    )
    asyncflow = await WorkflowEngine.create(execution_engine)

    # Build a tiny grid search space
    space = {
        "learning_rate": [0.001, 0.01],
        "num_layers": [1, 2],
    }

    GridRuntime = getattr(runtime_grid, "GridSearchRuntime",
                          getattr(runtime_grid, "GridRuntime"))
    runtime = GridRuntime(search_space=space, asyncflow=asyncflow)

    # Simple objective: no real ML; just deterministic function
    def objective(config):
        lr = config["learning_rate"]
        layers = config["num_layers"]
        return -abs(lr - 0.01) - 0.1 * abs(layers - 2)

    Learner = getattr(hpo_learner, "HpoLearner",
                      getattr(hpo_learner, "HPOLearner"))
    learner = Learner(runtime=runtime, objective=objective, max_trials=4)

    best_cfg, best_metric = await learner.run_async()

    assert best_cfg is not None
    assert isinstance(best_metric, (float, int))

    await asyncflow.shutdown()
