"""
Local HPO Example (Grid Search) using MNIST.

This script:
    1. Accepts hyperparameter search lists from the terminal.
    2. Loads and preprocesses the MNIST dataset.
    3. Defines an objective(hparams) that builds a Keras model.
    4. Runs a local (non-ROSE) grid search using HPOLearner.
"""

import argparse
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers

from rose.hpo import HPOLearner, HPOLearnerConfig


# ------------------------------------------------------------
#       1. Argument parsing (user inputs hyperparameters)
# ------------------------------------------------------------

def positive_int(value: str) -> int:
    # ensure integer >= 1
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
    parser = argparse.ArgumentParser(description="Local Grid Search HPO on MNIST")

    parser.add_argument("--learning_rate", nargs="+", type=positive_float, required=True)
    parser.add_argument("--num_layers",    nargs="+", type=positive_int,   required=True)
    parser.add_argument("--hidden_units",  nargs="+", type=positive_int,   required=True)
    parser.add_argument("--batch_size",    nargs="+", type=positive_int,   required=True)
    parser.add_argument("--l2_reg",        nargs="+", type=float,          required=True)

    return parser.parse_args()


args = parse_args()

# Build the parameter grid
param_grid = {
    "learning_rate": args.learning_rate,
    "num_layers":    args.num_layers,
    "hidden_units":  args.hidden_units,
    "batch_size":    args.batch_size,
    "l2_reg":        args.l2_reg,
}

config = HPOLearnerConfig(param_grid=param_grid, maximize=True)  # maximize accuracy


# ---------------------------------------------------------
# 2. Load MNIST
# ---------------------------------------------------------
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_val   = x_val.astype("float32") / 255.0

x_train = x_train.reshape((-1, 784))
x_val   = x_val.reshape((-1, 784))


# ---------------------------------------------------------
# 3. Objective function for MNIST model
# ---------------------------------------------------------
def objective_fn(hparams: dict) -> float:
    """
    Build and train a simple MLP on MNIST, returning validation accuracy.
    """

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(784,)))

    # Variable depth using num_layers
    for _ in range(hparams["num_layers"]):
        model.add(
            layers.Dense(
                hparams["hidden_units"],
                activation="relu",
                kernel_regularizer=regularizers.l2(hparams["l2_reg"])
            )
        )

    model.add(layers.Dense(10, activation="softmax"))

    optimizer = optimizers.Adam(hparams["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=hparams["batch_size"],
        epochs=5,
        verbose=0
    )

    val_acc = history.history["val_accuracy"][-1]
    print(f"Params {hparams} → Val Accuracy = {val_acc:.4f}")

    return float(val_acc)


# ---------------------------------------------------------
# 4. Run GRID SEARCH locally (no ROSE)
# ---------------------------------------------------------
learner = HPOLearner(objective_fn=objective_fn, config=config)
best_params, best_score, history = learner.grid_search()

print("\n================= GRID SEARCH RESULTS =================")
print(f"Total configurations tried: {len(history)}")
print(f"Best params: {best_params}")
print(f"Best accuracy: {best_score:.4f}")
print("========================================================")
