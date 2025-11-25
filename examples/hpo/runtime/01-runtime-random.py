"""
01-runtime-random.py

Distributed RANDOM SEARCH over MNIST hyperparameters using the ROSE runtime.

This script is the runtime-level counterpart of `01-local-random.py`:
- Local   examples: run HPO on a single machine (no asyncflow / ROSE).
- Runtime examples: submit each config as a separate ROSE task, executed
  via `radical.asyncflow` with a `ProcessPoolExecutor`.

Here we:
  * Load MNIST.
  * Define a simple MLP (no dropout, up to 2 dense layers).
  * Wrap training + evaluation in an `objective(params)` function.
  * Use `HPOLearner` + `HPOLearnerConfig` to manage hyperparameters.
  * Call `run_random_search_distributed` to evaluate a random subset
    of configs in parallel on the ROSE runtime.

Command-line interface mirrors `00-runtime-grid.py` and local examples.
"""

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger

from rose.hpo import HPOLearner, HPOLearnerConfig
from rose.hpo.runtime import run_random_search_distributed


# ---------------------------------------------------------------------------
# 1) Data utilities: load & preprocess MNIST
# ---------------------------------------------------------------------------

def load_mnist(normalize: bool = True):
    """
    Load the MNIST dataset and return (x_train, y_train), (x_val, y_val).

    We split the original training set into:
      - 50k samples for training
      - 10k samples for validation

    Shapes:
      x_train: (50000, 784)  float32
      y_train: (50000,)      int64
      x_val  : (10000, 784)  float32
      y_val  : (10000,)      int64
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten 28x28 images → 784-dim vectors
    x_train_full = x_train_full.reshape(-1, 784).astype("float32")
    x_test = x_test.reshape(-1, 784).astype("float32")

    if normalize:
        x_train_full /= 255.0
        x_test /= 255.0

    # Take 50k for training, 10k for validation
    x_train = x_train_full[:50000]
    y_train = y_train_full[:50000]
    x_val = x_train_full[50000:]
    y_val = y_train_full[50000:]

    return (x_train, y_train), (x_val, y_val)


# ---------------------------------------------------------------------------
# 2) Model builder: simple 1–2 layer MLP (no dropout)
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int,
              num_layers: int,
              hidden_units: int,
              learning_rate: float,
              l2_reg: float) -> tf.keras.Model:
    """
    Build a simple fully-connected network for MNIST classification.

    Constraints (aligned with project requirements):
      - No dropout.
      - 1 or 2 hidden layers.
      - ReLU activations.
      - Softmax output for 10 classes.
    """
    assert num_layers in (1, 2), "num_layers must be 1 or 2"

    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0.0 else None

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # First hidden layer
    model.add(
        layers.Dense(
            hidden_units,
            activation="relu",
            kernel_regularizer=regularizer,
        )
    )

    # Optional second hidden layer
    if num_layers == 2:
        model.add(
            layers.Dense(
                hidden_units,
                activation="relu",
                kernel_regularizer=regularizer,
            )
        )

    # Output layer: 10-way softmax
    model.add(layers.Dense(10, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# 3) Objective wrapper for HPOLearner: params -> val_accuracy (to maximize)
# ---------------------------------------------------------------------------

def make_objective(x_train: np.ndarray,
                   y_train: np.ndarray,
                   x_val: np.ndarray,
                   y_val: np.ndarray,
                   epochs: int):
    """
    Create an objective(params) -> score function.

    The score is validation accuracy after training on MNIST with the
    hyperparameters in `params`. Higher is better (maximize=True).
    """

    def objective(params: Dict[str, Any]) -> float:
        # Extract hyperparameters (already converted to Python scalars)
        learning_rate = float(params["learning_rate"])
        num_layers = int(params["num_layers"])
        hidden_units = int(params["hidden_units"])
        batch_size = int(params["batch_size"])
        l2_reg = float(params["l2_reg"])

        # Build and train the model
        model = build_mlp(
            input_dim=x_train.shape[1],
            num_layers=num_layers,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=0,
        )

        # Last epoch's validation accuracy
        val_acc = float(history.history["val_accuracy"][-1])

        print(
            f"[RUNTIME-RANDOM] Params {params} → "
            f"Val Accuracy = {val_acc:.4f}"
        )

        return val_acc  # we want to maximize this

    return objective


# ---------------------------------------------------------------------------
# 4) CLI parser: expose hyperparameter ranges + random search config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Command-line interface for configuring the random search.

    Example:

      python3 01-runtime-random.py \
        --learning_rate 0.0005 0.001 0.002 \
        --num_layers 1 2 \
        --hidden_units 64 128 \
        --batch_size 32 64 \
        --l2_reg 0.0 0.0001 \
        --epochs 3 \
        --n_samples 20 \
        --rng_seed 0
    """
    parser = argparse.ArgumentParser(
        description="Distributed random search over MNIST hyperparameters using ROSE runtime.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs="+",
        default=[0.0005, 0.001],
        help="Candidate learning rates.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Number of hidden layers (1 or 2).",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Number of units per hidden layer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Candidate batch sizes.",
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        nargs="+",
        default=[0.0, 0.0001],
        help="Candidate L2 regularization strengths.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs per configuration.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="How many random configurations to evaluate.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="Random seed for drawing configurations.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 5) Main: set up ROSE runtime and run distributed random search
# ---------------------------------------------------------------------------

async def main():
    # 5.1 Parse CLI flags
    args = parse_args()

    # 5.2 Initialize logging + execution backend + workflow engine
    init_default_logger()

    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend)

    try:
        # 5.3 Load MNIST data
        (x_train, y_train), (x_val, y_val) = load_mnist()

        # 5.4 Build objective function
        objective = make_objective(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=args.epochs,
        )

        # 5.5 Define discrete search space from CLI ranges
        param_grid: Dict[str, List[Any]] = {
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "hidden_units": args.hidden_units,
            "batch_size": args.batch_size,
            "l2_reg": args.l2_reg,
            # epochs is fixed per run and captured via closure
        }

        config = HPOLearnerConfig(
            param_grid=param_grid,
            maximize=True,  # we maximize validation accuracy
        )

        # Wrap objective in HPOLearner (shared abstraction with local examples)
        hpo = HPOLearner(objective_fn=objective, config=config)

        # 5.6 Run RANDOM SEARCH in a distributed fashion via ROSE runtime
        result = await run_random_search_distributed(
            asyncflow=asyncflow,
            hpo=hpo,
            n_samples=args.n_samples,
            rng_seed=args.rng_seed,
        )

        best_params = result["best_params"]
        best_score = result["best_score"]
        history = result["history"]

        print("\n=== HPO via ROSE runtime (RANDOM, distributed, MNIST) ===")
        print(f"Tried {len(history)} random configurations")
        print(f"Best params: {best_params}")
        print(f"Best score (Val Accuracy): {best_score:.4f}")

        print("\n=== Result object returned to main() ===")
        print(result)

    finally:
        # 5.7 Clean shutdown of the workflow engine + backend
        await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
