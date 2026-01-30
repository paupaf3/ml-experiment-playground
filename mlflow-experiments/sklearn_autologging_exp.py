import mlflow

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 42


def initialize_mlflow_config():
    mlflow.set_experiment("sklearn-autologging-train-ml-exp")

    # This enables system metrics - commonly used for deep learning
    mlflow.config.enable_system_metrics_logging()
    # Log every 1 seconds
    mlflow.config.set_system_metrics_sampling_interval(1)

    # Enable autologging for sklearn
    mlflow.sklearn.autolog()


def load_data():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    return X_train, X_test, y_train, y_test


def train_model(X, y):
    model = RandomForestClassifier(random_state=SEED)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    initialize_mlflow_config()
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    # validate_model()
