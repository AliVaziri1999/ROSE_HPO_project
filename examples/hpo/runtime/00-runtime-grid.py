"""
Runtime HPO Example (Grid Search) using MNIST + ROSE.

This script:
    1. Parses a hyperparameter grid from the command line.
    2. Defines an objective(params) that trains a Keras MLP on MNIST
       and returns validation accuracy.
    3. Wraps the search space in HPOLearnerConfig.
    4. Uses `run_grid_search_distributed` to evaluate all configs
       via the ROSE runtime (asyncflow + ProcessPoolExecutor).

Conceptually:
    - LOCAL examples (00-03 in examples/hpo/local) run everything
      inside a single Python process.

    - This RUNTIME example keeps the *same objective* and
      *same hyperparameters*, but executes each configuration as a
      distributed task managed by ROSE.
"""

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger

from rose.hpo import HPOLearner, HPOLearnerConfig
from rose.hpo.runtime import run_grid_search_distributed


# -------------------------------------------------------------------
# 1. CLI parsing utilities (same style as local examples)
# -------------------------------------------------------------------

def positive_int(value: str) -> int:
    """Ensure integer >= 1."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value}")
    return ivalue


def positive_float(value: str) -> float:
    """Ensure float > 0."""
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive float, got {value}")
    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed Grid Search HPO on MNIST via ROSE runtime"
    )

    # Same hyperparameters as in 00/01/02/03-local-*.py
    parser.add_argument("--learning_rate", nargs="+", type=positive_float, required=True)
    parser.add_argument("--num_layers",    nargs="+", type=positive_int,   required=True)
    parser.add_argument("--hidden_units",  nargs="+", type=positive_int,   required=True)
    parser.add_argument("--batch_size",    nargs="+", type=positive_int,   required=True)
    parser.add_argument("--l2_reg",        nargs="+", type=float,          required=True)

    # allow user to tweak epochs (default 5)
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=5,
        help="Number of training epochs per configuration (default: 5).",
    )

    # checkpoint arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="If set, save HPO history as JSON to this path periodically.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=positive_int,
        default=10,
        help="Save checkpoint every N successful evaluations (only if checkpoint_path is set).",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# 2. Objective function: MNIST MLP → validation accuracy
# -------------------------------------------------------------------

def mnist_objective(params: dict) -> float:
    """
    Objective function used by HPOLearner and executed by ROSE.

    Given a dict of hyperparameters, we:
      - build a simple MLP classifier for MNIST
      - train for a small fixed number of epochs
      - return the validation accuracy (float)

    NOTE:
        This function is defined at module top-level so that it can be
        pickled and executed in separate worker processes by the
        ProcessPoolExecutor used inside the ROSE runtime.
    """

    # Hyperparameters (cast to plain Python types if needed)
    learning_rate = float(params["learning_rate"])
    num_layers    = int(params["num_layers"])
    hidden_units  = int(params["hidden_units"])
    batch_size    = int(params["batch_size"])
    l2_reg        = float(params["l2_reg"])

    # allow passing "epochs" in params
    epochs = int(params.get("epochs", 5))

    # ---- Load and preprocess MNIST ----
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_val   = x_val.astype("float32") / 255.0

    # Flatten 28x28 images into 784D vectors
    x_train = x_train.reshape((-1, 784))
    x_val   = x_val.reshape((-1, 784))

    # ---- Build model ----
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(784,)))

    for _ in range(num_layers):
        model.add(
            layers.Dense(
                hidden_units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
            )
        )

    model.add(layers.Dense(10, activation="softmax"))

    optimizer = optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
    )

    val_acc = float(history.history["val_accuracy"][-1])
    print(f"[RUNTIME-GRID] Params {params} → Val Accuracy = {val_acc:.4f}")

    # For HPO, we MAXIMIZE this metric
    return val_acc


# -------------------------------------------------------------------
# 3. Async main: set up ROSE runtime and run distributed grid search
# -------------------------------------------------------------------

async def main(args):
    # 1) Configure ROSE / asyncflow runtime
    init_default_logger()

    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend)

    # 2) Build search space from CLI arguments
    search_space = {
        "learning_rate": args.learning_rate,
        "num_layers":    args.num_layers,
        "hidden_units":  args.hidden_units,
        "batch_size":    args.batch_size,
        "l2_reg":        args.l2_reg,
    }

    # Inject epochs as a "constant" hyperparameter if user specified it.
    if args.epochs:
        search_space["epochs"] = [args.epochs]

    config = HPOLearnerConfig(
        param_grid=search_space,
        maximize=True,  # we maximize accuracy now, not minimize RMSE
    )

    # 3) Create HPOLearner with MNIST objective
    hpo = HPOLearner(objective_fn=mnist_objective, config=config)

    # 4) Run **distributed** grid search via ROSE runtime
    result = await run_grid_search_distributed(
        asyncflow,
        hpo,
        checkpoint_path=args.checkpoint_path,
        checkpoint_freq=args.checkpoint_freq,
    )

    best_params = result["best_params"]
    best_score = result["best_score"]
    history    = result["history"]

    print("\n=== HPO via ROSE runtime (GRID, distributed, MNIST) ===")
    print(f"Best params: {best_params}")
    print(f"Best score (Val Accuracy): {best_score:.4f}")
    print(f"Total configs tried: {len(history)}")

    print("\n=== Result object returned to main() ===")
    print(result)

    # 5) Shutdown runtime cleanly
    await asyncflow.shutdown()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
