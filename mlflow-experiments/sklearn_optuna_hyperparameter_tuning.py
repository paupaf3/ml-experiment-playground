import mlflow
import optuna

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

SEED = 42


def initialize_mlflow_config():
    mlflow.set_experiment("sklearn-optuna-hyperparameter-tuning-ml-xp")

    # This enables system metrics - commonly used for deep learning
    mlflow.config.enable_system_metrics_logging()
    # Log every 1 seconds
    mlflow.config.set_system_metrics_sampling_interval(1)


def load_data():
    X, y = datasets.fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    error = mean_squared_error(y_val, y_pred)
    # Log current trial's error metric
    mlflow.log_metrics({"error": error})
    return error


def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int(
            "rf_n_estimators", 50, 400, log=True)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 1.0)

        params = {
            "max_depth": rf_max_depth,
            "n_estimators": rf_n_estimators,
            "max_features": rf_max_features,
            "random_state": SEED
        }

        # Log current trial's parameters
        mlflow.log_params(params)

        # Train and evaluate model
        regressor = RandomForestRegressor(**params)
        regressor.fit(X_train, y_train)
        error = evaluate_model(regressor, X_test, y_test)

        # Log the model file
        mlflow.sklearn.log_model(regressor, name="model")
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        return error


if __name__ == "__main__":
    initialize_mlflow_config()

    X_train, X_test, y_train, y_test = load_data()

    # Create a parent run that contains all child runs for different trials
    with mlflow.start_run(run_name="optuna-hyperparameter-tuning") as parent_run:
        # Log experiment settings
        n_trials = 50
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_error": study.best_value})

        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
