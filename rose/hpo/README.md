# ROSE HPO

This folder extends the ROSE toolkit with a centralized Hyperparameter Optimization (HPO) module.  
The HPO module (`rose/hpo`) provides a unified interface to run HPO strategies on top of ROSE and RADICAL-AsyncFlow:

- Grid search  
- Random search  
- Bayesian optimization  
- Genetic algorithm  

It is designed to launch multiple surrogate training runs in parallel on local or HPC resources, while keeping a shared history and best configuration across all strategies.

## HPO Code Structure

Main components inside `rose/hpo/`:

- `__init__.py` – package exports  
- `hpo_learner.py` – centralized HPOLearner and configuration  
- `manager.py` – task preparation and result collection  
- `runtime_grid.py`, `runtime_random.py`, `runtime_bayesian.py`, `runtime_genetic.py` – ROSE AsyncFlow runtime integrations  
- `bootstrap_hpo.py` – CLI bootstrapper used in examples and documentation

Example scripts are located in:

- `examples/hpo/` – local and runtime examples for all strategies

For more details, refer to `docs/ROSE_hpo_project.pdf` and `docs/Guide_to_run.pdf`.

## Installation (from repo root)

```
python3 -m venv rose_env
source rose_env/bin/activate

pip install -e .
```

This installs ROSE and the HPO extension in editable mode.

## Basic Usage (CLI Bootstrapper)

Run a distributed HPO job on the MNIST MLP surrogate:

Grid Search:
```
python -m rose.hpo.bootstrap_hpo \
  --strategy grid \
  --epochs 2 \
  --learning_rate 0.0005 0.001 \
  --num_layers 1 2 \
  --hidden_units 64 128 \
  --batch_size 64 \
  --l2_reg 0.0 0.0001 \
  --checkpoint_path grid_ckpt.json \
  --checkpoint_freq 4
```

Random Search:
```
python -m rose.hpo.bootstrap_hpo \
  --strategy random \
  --epochs 2 \
  --learning_rate 0.0005 0.001 \
  --num_layers 1 2 \
  --hidden_units 64 128 \
  --batch_size 64 \
  --l2_reg 0.0 0.0001 \
  --random_n_samples 8 \
  --random_rng_seed 0 \
  --checkpoint_path random_ckpt.json \
  --checkpoint_freq 4
```
Bayesian Optimization:
```
python -m rose.hpo.bootstrap_hpo \
  --strategy bayesian \
  --epochs 2 \
  --learning_rate 0.0005 0.001 \
  --num_layers 1 2 \
  --hidden_units 64 128 \
  --batch_size 64 \
  --l2_reg 0.0 0.0001 \
  --bayes_n_iter 8 \
  --bayes_init_points 4 \
  --bayes_rng_seed 0 \
  --checkpoint_path bayes_ckpt.json \
  --checkpoint_freq 4
```
Genetic Algorithm:
```
python -m rose.hpo.bootstrap_hpo \
  --strategy ga \
  --epochs 2 \
  --learning_rate 0.0005 0.001 \
  --num_layers 1 2 \
  --hidden_units 64 128 \
  --batch_size 64 \
  --l2_reg 0.0 0.0001 \
  --ga_population_size 6 \
  --ga_generations 4 \
  --ga_mutation_rate 0.1 \
  --ga_crossover_rate 0.5 \
  --ga_rng_seed 0 \
  --checkpoint_path ga_ckpt.json \
  --checkpoint_freq 4
```
