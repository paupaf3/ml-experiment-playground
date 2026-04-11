# ml-experiment-playground

A personal sandbox for building ML intuition from the ground up — starting with raw C/C++ implementations of core concepts, then scaling up to tracked deep learning experiments with MLflow.

## Purpose

This repo captures hands-on learning across:

1. **ML Foundations in C/C++** — implementing core ML primitives (loss functions, neurons, matrix ops) without any framework, to build a deep understanding of what happens under the hood.
2. **MLflow Experiment Tracking** — applying experiment tracking best practices to real training pipelines using PyTorch and scikit-learn.

---

## Contents

### ML Foundations in C (`ml-foundations-c/`)

Bare-metal ML concepts written in C. No libraries, no abstractions — just the math.

| File                                 | What it does                                                     |
| ------------------------------------ | ---------------------------------------------------------------- |
| `squared_error/squared_error_calc.c` | Interactive CLI tool to compute squared error `(pred - target)²` |
| `simple_neuron/simple_neuron.c`      | Single neuron with weighted inputs, bias, and activation function |

**Run any exercise:**

```bash
cd ml-foundations-c
./run.sh squared_error/squared_error_calc.c
```

---

### ML Foundations in C++ (`ml-foundations-cpp/`)

Same concepts ported to C++ to practice idiomatic C++ alongside ML fundamentals.

| File                                   | What it does                                                                |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `squared_error/squared_error_calc.cpp` | Interactive CLI tool to compute squared error, using `std::cin`/`std::cout` |
| `simple_neuron/simple_neuron.cpp`      | Single neuron with weighted inputs, bias, and activation function |
| `neural_network/neural_network.cpp`    | Multi-layer neural network built from scratch: forward pass, backpropagation, and weight updates |

**Run any exercise:**

```bash
cd ml-foundations-cpp
./run.sh squared_error/squared_error_calc.cpp
```

---

### MLflow Experiments (`mlflow-experiments/`)

A progression of MLflow experiment tracking patterns, from basic logging to full deep learning pipelines.

| Script                                       | Framework             | What it covers                                                                                                     |
| -------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `sklearn_logging_exp.py`                     | scikit-learn          | Manual parameter/metric/model logging; Logistic Regression on Iris                                                 |
| `sklearn_autologging_exp.py`                 | scikit-learn          | MLflow autologging; Random Forest classifier on Iris                                                               |
| `sklearn_optuna_hyperparameter_tuning.py`    | scikit-learn + Optuna | Nested runs for hyperparameter search; Random Forest regressor on California Housing                               |
| `pytorch_logging_train_exp.py`               | PyTorch               | Full training loop with per-batch and per-epoch metric logging; MLP on FashionMNIST; per-epoch model checkpointing |
| `pytorch_logging_inference_exp.py`           | PyTorch               | Loading a model from a previous MLflow run; logging final test metrics back to the same run                        |
| `pytorch_logging_inference_image_viz_exp.py` | PyTorch + Matplotlib  | Everything above + logging prediction visualizations (random samples & misclassified examples) as MLflow artifacts |

#### Setup

```bash
cd mlflow-experiments
pipenv install
pipenv shell
```

Run any experiment:

```bash
python sklearn_logging_exp.py
python pytorch_logging_train_exp.py
```

Launch the MLflow UI to inspect runs:

```bash
mlflow ui
```

#### Tech Stack

- **ML/DL:** PyTorch, scikit-learn
- **Experiment Tracking:** MLflow
- **Hyperparameter Tuning:** Optuna
- **Visualization:** Matplotlib
- **Environment:** Pipenv

---

## What I Learned

- How loss functions, neurons, and matrix operations work at the arithmetic level, implemented directly in C and C++
- MLflow's core tracking primitives: `log_params`, `log_metrics`, `log_model`, `log_figure`, `set_tag`
- The difference between manual logging and autologging, and when to use each
- How to structure nested parent/child runs for hyperparameter search with Optuna
- How to resume a previous MLflow run to append inference-time metrics and visualizations
- End-to-end PyTorch training loop with MLflow: per-batch metrics, epoch checkpoints, and final model artifact logging

---

## License

Personal learning repository — feel free to explore and adapt.
