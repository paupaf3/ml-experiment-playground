import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42


def initialize_mlflow_config():
    mlflow.set_experiment("pytorch-logging-dl-exp")

    # This enables system metrics - commonly used for deep learning
    mlflow.config.enable_system_metrics_logging()
    # Log every 1 seconds
    mlflow.config.set_system_metrics_sampling_interval(1)


def get_preprocess_pipeline():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform


def get_dataloaders(transform):
    train_dataset = datasets.FashionMNIST(
        "data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        "data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    initialize_mlflow_config()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data processing pipeline
    transform = get_preprocess_pipeline()

    loss_fn = nn.CrossEntropyLoss()

    # Get data loaders
    train_loader, test_loader = get_dataloaders(transform)

    # Load the final model of a previous expertiment
    run_id = "f88a34cfc7ef4c7480f7b7ea59743580"  # Obtained from mlflow UI
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/final_model")
    #  Or load checkpoint directly
    # model = mlflow.pytorch.load_model("runs:/<run_id>/checkpoint_<epoch>")
    model.to(device)
    model.eval()

    # Resume the previous run to log test metrics
    with mlflow.start_run(run_id=run_id) as run:
        # Evaluate the model on the test set
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = loss_fn(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        # Calculate and log final test metrics
        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * test_correct / test_total

        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
        print(f"Final Test Accuracy: {test_acc:.2f}%")
