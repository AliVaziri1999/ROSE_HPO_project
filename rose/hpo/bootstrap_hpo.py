"""
bootstrap_hpo.py

Unified CLI to run HPO experiments (GRID / RANDOM / BAYESIAN / GA)
on an MNIST MLP using the ROSE runtime.

Features:
  - Shared MNIST loader and 1–2 layer MLP (no dropout).
  - Unified hyperparameter space:
      * learning_rate
      * num_layers (1 or 2)
      * hidden_units
      * batch_size
      * l2_reg
      * epochs
  - Strategy selection:
      * GRID
      * RANDOM
      * BAYESIAN
      * GA (Genetic Algorithm)
  - Random Search knobs:
      * random_n_samples
      * random_rng_seed
  - Bayesian Search knobs:
      * bayes_n_init
      * bayes_n_iter
      * bayes_kappa
      * bayes_rng_seed
  - GA knobs:
      * population_size
      * n_generations
      * tournament_size
      * crossover_rate
      * mutation_rate
      * elitism
      * ga_rng_seed
  - Checkpointing:
      * checkpoint_path
      * checkpoint_freq
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

from .hpo_learner import HPOLearner, HPOLearnerConfig
from .runtime import (
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

def make_objective(x_train, y_train, x_val, y_val, epochs: int):
    """
    Create an objective(params) -> score function.

    The score is validation accuracy after training on MNIST with the
    hyperparameters in `params`. Higher is better (maximize=True).
    """

    def objective(params: Dict[str, Any]) -> float:
        # Extract hyperparameters (cast out of numpy scalars if needed)
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
            f"[BOOTSTRAP-{params.get('_strategy', 'HPO')}] "
            f"Params {params} → Val Accuracy = {val_acc:.4f}"
        )

        return val_acc  # maximize

    return objective


# ---------------------------------------------------------------------------
# 4) CLI parser: every knob for all strategies
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrapper for HPO over MNIST MLP using ROSE runtime "
                    "(GRID / RANDOM / BAYESIAN / GA)."
    )

    # ----- global: strategy + dataset / training -----
    p.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random", "bayesian", "ga"],
        default="grid",
        help="Which HPO strategy to run.",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs per configuration.",
    )

    p.add_argument(
        "--no_normalize",
        action="store_true",
        help="If set, do NOT normalize MNIST pixel values to [0,1].",
    )

    # ----- shared search space (applies to all strategies) -----
    p.add_argument(
        "--learning_rate",
        type=float,
        nargs="+",
        default=[0.0005, 0.001],
        help="Candidate learning rates.",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Number of hidden layers (1 or 2).",
    )
    p.add_argument(
        "--hidden_units",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Number of units per hidden layer.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Candidate batch sizes.",
    )
    p.add_argument(
        "--l2_reg",
        type=float,
        nargs="+",
        default=[0.0, 0.0001],
        help="Candidate L2 regularization strengths.",
    )

    # ----- RANDOM search knobs -----
    p.add_argument(
        "--random_n_samples",
        type=int,
        default=20,
        help="RANDOM: how many random configs to evaluate.",
    )
    p.add_argument(
        "--random_rng_seed",
        type=int,
        default=0,
        help="RANDOM: RNG seed for sampling configurations.",
    )

    # ----- BAYESIAN search knobs -----
    p.add_argument(
        "--bayes_n_init",
        type=int,
        default=5,
        help="BAYESIAN: number of initial random evaluations.",
    )
    p.add_argument(
        "--bayes_n_iter",
        type=int,
        default=15,
        help="BAYESIAN: number of BO iterations after init phase.",
    )
    p.add_argument(
        "--bayes_kappa",
        type=float,
        default=2.0,
        help="BAYESIAN: exploration parameter for UCB/LCB acquisition.",
    )
    p.add_argument(
        "--bayes_rng_seed",
        type=int,
        default=0,
        help="BAYESIAN: RNG seed for BO.",
    )

    # ----- GA (Genetic Algorithm) knobs -----
    p.add_argument(
        "--ga_population_size",
        type=int,
        default=20,
        help="GA: population size.",
    )
    p.add_argument(
        "--ga_n_generations",
        type=int,
        default=15,
        help="GA: number of generations.",
    )
    p.add_argument(
        "--ga_tournament_size",
        type=int,
        default=3,
        help="GA: tournament size for selection.",
    )
    p.add_argument(
        "--ga_crossover_rate",
        type=float,
        default=0.9,
        help="GA: crossover probability.",
    )
    p.add_argument(
        "--ga_mutation_rate",
        type=float,
        default=0.2,
        help="GA: mutation probability.",
    )
    p.add_argument(
        "--ga_elitism",
        type=int,
        default=1,
        help="GA: number of elite individuals carried over.",
    )
    p.add_argument(
        "--ga_rng_seed",
        type=int,
        default=0,
        help="GA: RNG seed.",
    )

    # ----- checkpointing (for all runtime strategies) -----
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a JSON file where HPO history will be "
             "periodically saved. If not provided, checkpointing is disabled.",
    )
    p.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Save a checkpoint every N successful evaluations. "
             "If <= 0, checkpointing is disabled.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# 5) Main: set up ROSE runtime and run chosen strategy
# ---------------------------------------------------------------------------

async def _run_with_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Internal helper that:
      - initializes AsyncFlow backend,
      - runs the selected HPO strategy,
      - shuts everything down cleanly,
      - returns the result dict.
    """
    init_default_logger()

    backend = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(backend)

    try:
        # Load data
        (x_train, y_train), (x_val, y_val) = load_mnist(
            normalize=not args.no_normalize
        )

        # Build objective
        objective = make_objective(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=args.epochs,
        )

        # Shared search space
        param_grid: Dict[str, List[Any]] = {
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "hidden_units": args.hidden_units,
            "batch_size": args.batch_size,
            "l2_reg": args.l2_reg,
        }

        config = HPOLearnerConfig(
            param_grid=param_grid,
            maximize=True,
        )
        hpo = HPOLearner(objective_fn=objective, config=config)

        # Strategy dispatch
        if args.strategy == "grid":
            print("\n[BOOTSTRAP] Running GRID search (distributed via ROSE)...\n")
            result = await run_grid_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        elif args.strategy == "random":
            print("\n[BOOTSTRAP] Running RANDOM search (distributed via ROSE)...\n")
            result = await run_random_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                n_samples=args.random_n_samples,
                rng_seed=args.random_rng_seed,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        elif args.strategy == "bayesian":
            print("\n[BOOTSTRAP] Running BAYESIAN search (distributed via ROSE)...\n")
            result = await run_bayesian_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                n_init=args.bayes_n_init,
                n_iter=args.bayes_n_iter,
                kappa=args.bayes_kappa,
                rng_seed=args.bayes_rng_seed,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        elif args.strategy == "ga":
            print("\n[BOOTSTRAP] Running GA search (distributed via ROSE)...\n")
            result = await run_genetic_search_distributed(
                asyncflow=asyncflow,
                hpo=hpo,
                population_size=args.ga_population_size,
                n_generations=args.ga_n_generations,
                tournament_size=args.ga_tournament_size,
                crossover_rate=args.ga_crossover_rate,
                mutation_rate=args.ga_mutation_rate,
                elitism=args.ga_elitism,
                rng_seed=args.ga_rng_seed,
                checkpoint_path=args.checkpoint_path,
                checkpoint_freq=args.checkpoint_freq,
            )

        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        return result

    finally:
        await asyncflow.shutdown()


def main():
    args = parse_args()

    result = asyncio.run(_run_with_runtime(args))

    best_params = result["best_params"]
    best_score = result["best_score"]
    history = result["history"]

    print("\n=== HPO via ROSE runtime (bootstrap) ===")
    print(f"Strategy        : {args.strategy.upper()}")
    print(f"Tried configs   : {len(history)}")
    print(f"Best params     : {best_params}")
    print(f"Best score      : {best_score:.4f}")
    if args.checkpoint_path:
        print(f"Checkpoint file : {args.checkpoint_path}")
    print("========================================\n")


if __name__ == "__main__":
    main()
