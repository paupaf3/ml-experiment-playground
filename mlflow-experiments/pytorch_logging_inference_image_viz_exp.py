import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

SEED = 42

# FashionMNIST class names
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def visualize_predictions(images, predictions, targets, num_samples=16):
    """
    Create a figure comparing predicted vs actual labels for sample images.
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.ceil(num_samples / 4))
    cols = min(4, num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        img = images[i].cpu().numpy().squeeze()
        pred = predictions[i].item()
        actual = targets[i].item()

        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")

        # Color code: green if correct, red if wrong
        color = "green" if pred == actual else "red"
        axes[i].set_title(
            f"Pred: {CLASS_NAMES[pred]}\nActual: {CLASS_NAMES[actual]}",
            fontsize=9,
            color=color
        )

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


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
        all_images, all_predictions, all_targets = [], [], []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = loss_fn(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

                # Store samples for visualization
                all_images.append(data)
                all_predictions.append(predicted)
                all_targets.append(target)

        # Calculate and log final test metrics
        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * test_correct / test_total

        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
        print(f"Final Test Accuracy: {test_acc:.2f}%")

        # Concatenate all batches
        all_images = torch.cat(all_images, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Log random sample predictions visualization
        random_indices = torch.randperm(len(all_images))[:16]
        fig_random = visualize_predictions(
            all_images[random_indices],
            all_predictions[random_indices],
            all_targets[random_indices],
            num_samples=16
        )
        mlflow.log_figure(fig_random, "predictions/random_samples.png")
        plt.close(fig_random)

        # Log misclassified samples visualization
        misclassified_mask = all_predictions != all_targets
        misclassified_indices = torch.where(misclassified_mask)[0]

        if len(misclassified_indices) > 0:
            num_misclassified = min(16, len(misclassified_indices))
            fig_errors = visualize_predictions(
                all_images[misclassified_indices[:num_misclassified]],
                all_predictions[misclassified_indices[:num_misclassified]],
                all_targets[misclassified_indices[:num_misclassified]],
                num_samples=num_misclassified
            )
            mlflow.log_figure(
                fig_errors, "predictions/misclassified_samples.png")
            plt.close(fig_errors)
            print(f"Logged {num_misclassified} misclassified samples")

        print("Prediction visualizations logged to MLflow!")
