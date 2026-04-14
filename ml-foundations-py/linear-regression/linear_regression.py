
# Import numpy for numerical operations
import numpy as np

# --- Linear Regression from Scratch ---
# Model: y = w * x + b
# Where:
#   y = predicted value
#   x = input feature
#   w = weight (slope)
#   b = bias (intercept)
#
# The goal is to find w and b that minimize the mean squared error between predictions and true values.

# Example dataset: x (input), y (target)
x_all = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_all = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])


# Predict function: computes y = w * x + b for all x in X
def predict(X: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * X + b


# Compute mean squared error (MSE) between predictions and true values
# MSE = (1/n) * sum((y_pred - y_true)^2)
def compute_cost(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


# Compute gradients of the cost function with respect to w and b
# dw = d(MSE)/dw, db = d(MSE)/db
# These gradients tell us how to update w and b to reduce the error
def compute_gradients(X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    dw = float(2 * np.mean((y_pred - y_true) * X))  # Gradient w.r.t. w
    db = float(2 * np.mean(y_pred - y_true))        # Gradient w.r.t. b
    return dw, db


# Training function: uses gradient descent to optimize w and b
# Steps:
#   1. Initialize w and b
#   2. For a number of iterations:
#       a. Predict y using current w, b
#       b. Compute cost (MSE)
#       c. Compute gradients
#       d. Update w and b using gradients and learning rate
#   3. Return the learned w and b
def train(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    w = 0.0  # Initial weight
    b = 0.0  # Initial bias
    learning_rate = 0.01  # Step size for each update
    num_iterations = 5000  # Number of times to update w and b

    for i in range(num_iterations):
        y_pred = predict(X, w, b)
        cost = compute_cost(y_pred, y)
        dw, db = compute_gradients(X, y_pred, y)

        w -= learning_rate * dw
        b -= learning_rate * db

        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}, w {w:.4f}, b {b:.4f}")

    return w, b


# Evaluate the model on test data
# Prints the cost and learned parameters
def evaluate(X: np.ndarray, y: np.ndarray, w: float, b: float) -> None:
    y_pred = predict(X, w, b)
    cost = compute_cost(y_pred, y)
    print(f"Cost: {cost:.4f}, w: {w:.4f}, b: {b:.4f}")


# Main script: trains and evaluates the linear regression model
if __name__ == "__main__":

    # Split data into training and testing sets (80% train, 20% test)
    split_index = int(0.8 * len(x_all))
    x_train, y_train = x_all[:split_index], y_all[:split_index]
    x_test, y_test = x_all[split_index:], y_all[split_index:]

    # Train the model on training data
    w, b = train(x_train, y_train)

    # Evaluate the model on test data
    evaluate(x_test, y_test, w, b)
