"""
02-runtime-bayesian.py

Bayesian Optimization over MNIST hyperparameters.

This script:
  - builds a MNIST MLP (no dropout, 1–2 hidden layers)
  - defines an objective = validation accuracy
  - creates an HPOLearner + HPOLearnerConfig
  - runs Bayesian Optimization via run_bayesian_hpo_task()

If --distributed is given, the BO evaluations are executed using the
ROSE / AsyncFlow runtime (see rose.hpo.runtime_bayesian).
Otherwise, the optimization is performed locally.
"""

import argparse
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from rose.hpo import HPOLearner, HPOLearnerConfig, run_bayesian_hpo_task


# ---------------------------------------------------------------------------
# 1) MNIST DATA
# ---------------------------------------------------------------------------

def load_mnist(normalize: bool = True):
    """
    Load and flatten MNIST, then split:
      - 50k for training
      - 10k for validation
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
# 2) MODEL: 1–2 hidden layers, NO DROPOUT
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int,
              num_layers: int,
              hidden_units: int,
              learning_rate: float,
              l2_reg: float) -> tf.keras.Model:
    """
    Simple fully-connected classifier for MNIST, satisfying project rules:
      - No dropout.
      - 1 or 2 hidden layers.
      - ReLU activations.
      - Final 10-way softmax.
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

    # Output layer
    model.add(layers.Dense(10, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 3) OBJECTIVE(params) → validation accuracy (maximize)
# ---------------------------------------------------------------------------

def make_objective(x_train,
                   y_train,
                   x_val,
                   y_val,
                   epochs: int):

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
        print(f"[RUNTIME-BO] Params {params} → Val Accuracy = {val_acc:.4f}")
        return val_acc  # maximize

    return objective


# ---------------------------------------------------------------------------
# 4) CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Example:

      python -m examples.hpo.runtime.02-runtime-bayesian \
        --learning_rate 0.0005 0.001 0.002 \
        --num_layers 1 2 \
        --hidden_units 64 128 \
        --batch_size 32 64 \
        --l2_reg 0.0 0.0001 \
        --epochs 3 \
        --n_init 4 \
        --n_iter 8 \
        --kappa 2.0 \
        --rng_seed 0 \
        --distributed
    """
    parser = argparse.ArgumentParser(
        description="Bayesian optimization over MNIST hyperparameters."
    )

    parser.add_argument("--learning_rate", type=float, nargs="+",
                        default=[0.0005, 0.001, 0.002])
    parser.add_argument("--num_layers", type=int, nargs="+",
                        default=[1, 2])
    parser.add_argument("--hidden_units", type=int, nargs="+",
                        default=[64, 128])
    parser.add_argument("--batch_size", type=int, nargs="+",
                        default=[32, 64])
    parser.add_argument("--l2_reg", type=float, nargs="+",
                        default=[0.0, 0.0001])

    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per evaluation.")

    parser.add_argument("--n_init", type=int, default=4,
                        help="Initial random evaluations before BO.")
    parser.add_argument("--n_iter", type=int, default=8,
                        help="Number of BO iterations.")
    parser.add_argument("--kappa", type=float, default=2.0,
                        help="UCB exploration parameter.")
    parser.add_argument("--rng_seed", type=int, default=0,
                        help="Random seed for BO / HPOLearner.")

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="If set, run Bayesian optimization using ROSE / AsyncFlow.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 5) MAIN – local or distributed BO
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 5.1 Load data
    (x_train, y_train), (x_val, y_val) = load_mnist()

    # 5.2 Objective
    objective = make_objective(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
    )

    # 5.3 Search space
    param_grid = {
        "learning_rate": args.learning_rate,
        "num_layers": args.num_layers,
        "hidden_units": args.hidden_units,
        "batch_size": args.batch_size,
        "l2_reg": args.l2_reg,
    }

    config = HPOLearnerConfig(
        param_grid=param_grid,
        maximize=True,  # maximize validation accuracy
    )

    hpo = HPOLearner(
        objective_fn=objective,
        config=config,
    )

    # 5.4 Run Bayesian optimization (local or distributed)
    result = run_bayesian_hpo_task(
        hpo=hpo,
        n_init=args.n_init,
        n_iter=args.n_iter,
        kappa=args.kappa,
        rng_seed=args.rng_seed,
        distributed=args.distributed,
    )

    print("\n===== BAYESIAN OPTIMIZATION RESULT =====")
    print(f"Total evaluations: {len(result['history'])}")
    print(f"Best params: {result['best_params']}")
    print(f"Best Val Accuracy: {result['best_score']:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
