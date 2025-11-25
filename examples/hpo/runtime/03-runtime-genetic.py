import argparse
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from rose.hpo import HPOLearner, HPOLearnerConfig
from rose.hpo.runtime_genetic import run_genetic_hpo_task


# ----------------------------------------------------------
#                       MNIST utilities
# ----------------------------------------------------------

def load_mnist(normalize: bool = True):
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


# ----------------------------------------------------------
#                       MLP model builder
# ----------------------------------------------------------

def build_mlp(input_dim, num_layers, hidden_units, learning_rate, l2_reg):
    assert num_layers in (1, 2)

    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    model.add(layers.Dense(hidden_units, activation="relu", kernel_regularizer=regularizer))

    if num_layers == 2:
        model.add(layers.Dense(hidden_units, activation="relu", kernel_regularizer=regularizer))

    model.add(layers.Dense(10, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------------------------------------------
#               Objective wrapper for HPOLearner
# ----------------------------------------------------------

def make_objective(x_train, y_train, x_val, y_val, epochs):
    def objective(params: Dict[str, Any]) -> float:
        lr = float(params["learning_rate"])
        nl = int(params["num_layers"])
        hu = int(params["hidden_units"])
        bs = int(params["batch_size"])
        l2 = float(params["l2_reg"])

        model = build_mlp(
            input_dim=x_train.shape[1],
            num_layers=nl,
            hidden_units=hu,
            learning_rate=lr,
            l2_reg=l2,
        )

        hist = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=bs,
            epochs=epochs,
            verbose=0
        )

        val_acc = float(hist.history["val_accuracy"][-1])
        print(f"[RUNTIME-GA] Params {params} → Val Accuracy = {val_acc:.4f}")
        return val_acc  # maximize

    return objective


# ----------------------------------------------------------
#                           CLI parser
# ----------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Genetic Algorithm HPO for MNIST (local or ROSE-distributed)."
    )

    # HPO search space
    p.add_argument("--learning_rate", type=float, nargs="+",
                   default=[0.0005, 0.001, 0.002])
    p.add_argument("--num_layers", type=int, nargs="+",
                   default=[1, 2])
    p.add_argument("--hidden_units", type=int, nargs="+",
                   default=[64, 128])
    p.add_argument("--batch_size", type=int, nargs="+",
                   default=[32, 64])
    p.add_argument("--l2_reg", type=float, nargs="+",
                   default=[0.0, 0.0001])

    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs per evaluation.")

    # GA hyperparameters
    p.add_argument("--population_size", type=int, default=20)
    p.add_argument("--n_generations", type=int, default=15)
    p.add_argument("--tournament_size", type=int, default=3)
    p.add_argument("--crossover_rate", type=float, default=0.9)
    p.add_argument("--mutation_rate", type=float, default=0.2)
    p.add_argument("--elitism", type=int, default=1)

    p.add_argument("--rng_seed", type=int, default=0)

    p.add_argument(
        "--distributed",
        action="store_true",
        help="If set, run GA using ROSE / AsyncFlow distributed evaluation.",
    )

    return p.parse_args()


# ----------------------------------------------------------
#               Main – local or distributed GA
# ----------------------------------------------------------

def main():
    args = parse_args()

    (x_train, y_train), (x_val, y_val) = load_mnist()

    objective = make_objective(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
    )

    space = {
        "learning_rate": args.learning_rate,
        "num_layers": args.num_layers,
        "hidden_units": args.hidden_units,
        "batch_size": args.batch_size,
        "l2_reg": args.l2_reg,
    }

    config = HPOLearnerConfig(param_grid=space, maximize=True)
    hpo = HPOLearner(objective_fn=objective, config=config)

    result = run_genetic_hpo_task(
        hpo=hpo,
        population_size=args.population_size,
        n_generations=args.n_generations,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        rng_seed=args.rng_seed,
        distributed=args.distributed,
    )

    print("\n=== GA HPO RESULT ===")
    print(f"Best params: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Total evaluations: {len(result['history'])}")
    print("======================\n")


if __name__ == "__main__":
    main()
