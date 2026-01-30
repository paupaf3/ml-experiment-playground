import mlflow

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

SEED = 42


def initialize_mlflow_config():
    mlflow.set_experiment("sklearn-logging-ml-exp")

    # This enables system metrics - commonly used for deep learning
    mlflow.config.enable_system_metrics_logging()
    # Log every 1 seconds
    mlflow.config.set_system_metrics_sampling_interval(1)


def load_data():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    return X_train, X_test, y_train, y_test


def train_model(X, y, params):
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", float(accuracy))


if __name__ == "__main__":
    initialize_mlflow_config()

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": SEED,
    }

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Train the model
        model = train_model(X_train, y_train, params)

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris-logistic-regression-model"
        )

        # Predict on the test set, compute and log the loss metric
        evaluate_model(model, X_test, y_test)

        # Optional: Set a tag that we can use to remind ourselves what this run was about
        mlflow.set_tag("Training info",
                       "Basic LR model on Iris dataset for test logging")
