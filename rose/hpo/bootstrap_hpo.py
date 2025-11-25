"""
Single entrypoint (bootstrapper) for running all ROSE-based distributed
HPO strategies on MNIST:

  --strategy {grid, random, bayesian, ga}

It reuses the same model and data pattern as the other runtime examples:
  * MNIST MLP (1-2 dense layers, ReLU + softmax)
  * HPOLearner + HPOLearnerConfig
  * Distributed evaluation via rose.hpo.runtime.* functions
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
from rose.hpo.runtime import (
    run_grid_search_distributed,
    run_random_search_distributed,
    run_bayesian_search_distributed,
    run_genetic_search_distributed,
)

# ---------------------------------------------------------------------------
# 1) Data utilities: load & preprocess MNIST
# ---------------------------------------------------------------------------


def load_mnist(normalize: bool = True):
    """
    Load the MNIST dataset and split it into:
      - 50k train
      - 10k validation

    Returns
    -------
    (x_train, y_train), (x_val, y_val)
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train_full = x_train_full.reshape(-1, 784).astype("float32")
    x_test = x_test.reshape(-1, 784).astype("float32")

    if normalize:
        x_train_full /= 255.0
        x_test /= 255.0

    x_train = x_train_full[:50000]
    y_train = y_train_full[:50000]
    x_val = x_train_full[50000:]
    y_val = y_train_full[50000:]

    return (x_train, y_train), (x_val, y_val)


# ---------------------------------------------------------------------------
# 2) Model builder: simple 1–2 layer MLP (no dropout)
# ---------------------------------------------------------------------------


def build_mlp(
    input_dim: int,
    num_layers: int,
    hidden_units: int,
    learning_rate: float,
    l2_reg: float,
) -> tf.keras.Model:
    """
    Build a simple fully-connected network for MNIST classification.

    Constraints:
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

    # second hidden layer
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
# 3) Objective wrapper for HPOLearner: params -> val_accuracy (maximize)
# ---------------------------------------------------------------------------


def make_objective(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
):
    """
    Create an objective(params) -> score function.

    The score is validation accuracy after training on MNIST with the
    hyperparameters in `params`. Higher is better (maximize=True).
    """

    def objective(params: Dict[str, Any]) -> float:
        learning_rate = float(params["learning_rate"])
        num_layers = int(params["num_layers"])
        hidden_units = int(params["hidden_units"])
        batch_size = int(params["batch_size"])
        l2_reg = float(params["l2_reg"])

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

        val_acc = float(history.history["val_accuracy"][-1])

        print(
            f"[BOOTSTRAP-{params.get('strategy', 'HPO').upper()}] "
            f"Params {params} → Val Accuracy = {val_acc:.4f}"
        )

        return val_acc

    return objective


# ---------------------------------------------------------------------------
# 4) CLI parser: common + strategy-specific options
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrapper for distributed HPO on MNIST using ROSE runtime.\n"
            "Choose a strategy with --strategy and configure hyperparameters."
        )
    )

    # Which HPO strategy to run
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random", "bayesian", "ga"],
        required=True,
        help="Which HPO strategy to use.",
    )

    # Common search-space hyperparameters
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
        help="Number of units in each hidden layer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Candidate batch sizes.",
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        nargs="+",
        default=[0.0, 0.001],
        help="Candidate L2 regularization strengths.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs per evaluation.",
    )

    # Checkpointing (applies to all strategies)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "If set, periodically save JSON history to this path "
            "(e.g., checkpoints/random_bootstrap.json)."
        ),
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=2,
        help="Checkpoint every N successful evaluations (<=0 disables).",
    )

    # Random search parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="[RANDOM] Number of random configurations to evaluate.",
    )

    # Bayesian search parameters
    parser.add_argument(
        "--n_init",
        type=int,
        default=5,
        help="[BAYESIAN] Number of initial random points.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=15,
        help="[BAYESIAN] Number of BO iterations after the initial points.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="[BAYESIAN] Exploration/exploitation trade-off constant.",
    )

    # GA parameters
    parser.add_argument(
        "--population_size",
        type=int,
        default=20,
        help="[GA] Population size.",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=15,
        help="[GA] Number of generations.",
    )
    parser.add_argument(
        "--tournament_size",
        type=int,
        default=3,
        help="[GA] Tournament size for selection.",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.9,
        help="[GA] Crossover probability.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.2,
        help="[GA] Mutation probability.",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=1,
        help="[GA] Number of elite individuals to preserve each generation.",
    )

    # Shared random seed
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="[RANDOM/BAYESIAN/GA] Random seed.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 5) Main: set up ROSE runtime and dispatch to selected strategy
# ---------------------------------------------------------------------------


async def main():
    args = parse_args()

    # Initialize ROSE logging + execution backend + workflow engine
    init_default_logger()
    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend)

    try:
        # Load data
        (x_train, y_train), (x_val, y_val) = load_mnist()

        # Build objective function (shared)
        objective = make_objective(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=args.epochs,
        )

        # Build discrete search space from CLI ranges
        param_grid: Dict[str, List[Any]] = {
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "hidden_units": args.hidden_units,
            "batch_size": args.batch_size,
            "l2_reg": args.l2_reg,
        }

        config = HPOLearnerConfig(
            param_grid=param_grid,
            maximize=True,  # we maximize validation accuracy
        )
        hpo = HPOLearner(objective_fn=objective, config=config)

        # Dispatch based on strategy
        strategy = args.strategy.lower()

        if strategy == "grid":
            print("\n[BOOTSTRAP] Running GRID search (distributed)...\n")
            result = await run_grid_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        elif strategy == "random":
            print("\n[BOOTSTRAP] Running RANDOM search (distributed)...\n")
            result = await run_random_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                n_samples=args.n_samples,
                rng_seed=args.rng_seed,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        elif strategy == "bayesian":
            print("\n[BOOTSTRAP] Running BAYESIAN optimization (distributed)...\n")
            result = await run_bayesian_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                n_init=args.n_init,
                n_iter=args.n_iter,
                kappa=args.kappa,
                rng_seed=args.rng_seed,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        elif strategy == "ga":
            print("\n[BOOTSTRAP] Running GENETIC ALGORITHM search (distributed)...\n")
            result = await run_genetic_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                population_size=args.population_size,
                n_generations=args.n_generations,
                tournament_size=args.tournament_size,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                elitism=args.elitism,
                rng_seed=args.rng_seed,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        best_params = result["best_params"]
        best_score = result["best_score"]
        history = result["history"]

        print("\n=== HPO via ROSE runtime (BOOTSTRAP) ===")
        print(f"Strategy: {strategy}")
        print(f"Tried {len(history)} configurations")
        print(f"Best params: {best_params}")
        print(f"Best score (Val Accuracy): {best_score:.4f}")
        print("========================================\n")

    finally:
        # Clean shutdown of the workflow engine + backend
        await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
