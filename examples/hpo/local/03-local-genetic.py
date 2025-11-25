"""
Local HPO Example (Genetic Algorithm) using MNIST.

This script:
    1. Accepts hyperparameter search lists from the terminal.
    2. Loads and preprocesses the MNIST dataset.
    3. Defines an objective(hparams) that builds a Keras model.
    4. Runs a local (non-ROSE) Genetic Algorithm search using HPOLearner.genetic_search().
"""

import argparse
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers

from rose.hpo import HPOLearner, HPOLearnerConfig


# ------------------------------------------------------------
#       1. Argument parsing (user inputs hyperparameters)
# ------------------------------------------------------------

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


def float_0_1(value: str) -> float:
    """Ensure 0.0 <= float <= 1.0 (for probabilities)."""
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(
            f"Expected float in [0, 1] for probability, got {value}"
        )
    return fvalue


def parse_args():
    parser = argparse.ArgumentParser(description="Local Genetic HPO on MNIST")

    # Same HPO space as other local examples
    parser.add_argument("--learning_rate", nargs="+", type=positive_float, required=True)
    parser.add_argument("--num_layers",    nargs="+", type=positive_int,   required=True)
    parser.add_argument("--hidden_units",  nargs="+", type=positive_int,   required=True)
    parser.add_argument("--batch_size",    nargs="+", type=positive_int,   required=True)
    parser.add_argument("--l2_reg",        nargs="+", type=float,          required=True)

    # Genetic Algorithm hyperparameters (Option A: fully exposed)
    parser.add_argument(
        "--population_size",
        type=positive_int,
        default=20,
        help="Population size (default: 20).",
    )
    parser.add_argument(
        "--n_generations",
        type=positive_int,
        default=30,
        help="Number of generations (default: 30).",
    )
    parser.add_argument(
        "--tournament_size",
        type=positive_int,
        default=3,
        help="Tournament size for selection (default: 3).",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float_0_1,
        default=0.9,
        help="Crossover probability in [0,1] (default: 0.9).",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float_0_1,
        default=0.1,
        help="Mutation probability in [0,1] (default: 0.1).",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="Random seed for GA (default: 0).",
    )

    return parser.parse_args()


args = parse_args()

# Build the parameter grid (discrete search space)
param_grid = {
    "learning_rate": args.learning_rate,
    "num_layers":    args.num_layers,
    "hidden_units":  args.hidden_units,
    "batch_size":    args.batch_size,
    "l2_reg":        args.l2_reg,
}

# We want to maximize validation accuracy
config = HPOLearnerConfig(param_grid=param_grid, maximize=True)


# ---------------------------------------------------------
# 2. Load MNIST
# ---------------------------------------------------------
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_val   = x_val.astype("float32") / 255.0

# Flatten 28x28 images into 784D vectors
x_train = x_train.reshape((-1, 784))
x_val   = x_val.reshape((-1, 784))


# ---------------------------------------------------------
# 3. Objective function for MNIST model
# ---------------------------------------------------------
def objective_fn(hparams: dict) -> float:
    """
    Build and train a simple MLP on MNIST, returning validation accuracy.

    hparams dict contains:
        - "learning_rate"
        - "num_layers"
        - "hidden_units"
        - "batch_size"
        - "l2_reg"
    """

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(784,)))

    # Variable depth using num_layers
    for _ in range(hparams["num_layers"]):
        model.add(
            layers.Dense(
                hparams["hidden_units"],
                activation="relu",
                kernel_regularizer=regularizers.l2(hparams["l2_reg"]),
            )
        )

    # Output layer for 10 classes
    model.add(layers.Dense(10, activation="softmax"))

    optimizer = optimizers.Adam(hparams["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train for a small, fixed number of epochs (fast example)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=hparams["batch_size"],
        epochs=5,
        verbose=0,
    )

    val_acc = float(history.history["val_accuracy"][-1])
    print(f"[GA] Params {hparams} → Val Accuracy = {val_acc:.4f}")

    return val_acc


# ---------------------------------------------------------
# 4. Run GENETIC ALGORITHM locally (no ROSE)
# ---------------------------------------------------------
learner = HPOLearner(objective_fn=objective_fn, config=config)

best_params, best_score, history = learner.genetic_search(
    population_size=args.population_size,
    n_generations=args.n_generations,
    tournament_size=args.tournament_size,
    crossover_rate=args.crossover_rate,
    mutation_rate=args.mutation_rate,
    rng_seed=args.rng_seed,
)

print("\n============= GENETIC HPO RESULTS =============")
print(f"Total evaluations (len(history)): {len(history)}")
print(f"Best params: {best_params}")
print(f"Best accuracy: {best_score:.4f}")
print("===============================================")
