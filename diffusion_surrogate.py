
import numpy as np


def predict_surrogate(X, W1, b1, W2, b2, X_mean, X_std):
    """Predict diffusion width from physical parameters."""

    # Normalize inputs
    X_scaled = (X - X_mean) / X_std

    # Forward propagation
    hidden = np.tanh(X_scaled @ W1 + b1)

    predictions = hidden @ W2 + b2

    return predictions.reshape(-1)


rng = np.random.default_rng(42)

# Generate simulation parameters
n_samples = 500

alpha = rng.uniform(0.01, 0.2, n_samples)
time = rng.uniform(0.1, 5.0, n_samples)

sigma0 = 0.2

# Analytical diffusion solution
sigma_squared = sigma0**2 + 2 * alpha * time
sigma = np.sqrt(sigma_squared)

# Model inputs: diffusion coefficient and time
X = np.column_stack((alpha, time))

# Model target: resulting profile width
y = sigma.reshape(-1, 1)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst five examples:")

for i in range(5):
    print(
        f"alpha={alpha[i]:.4f}, "
        f"time={time[i]:.4f}, "
        f"sigma={sigma[i]:.4f}"
    )



# Split dataset into training and testing examples
indices = rng.permutation(n_samples)

split = int(0.8 * n_samples)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

# Scale inputs using training data only
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))



# Network architecture: 2 -> 16 -> 1
hidden_size = 16

W1 = rng.normal(0, 0.3, (2, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = rng.normal(0, 0.3, (hidden_size, 1))
b2 = np.zeros((1, 1))

learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    z1 = X_train @ W1 + b1
    a1 = np.tanh(z1)

    predictions = a1 @ W2 + b2

    # Calculate loss
    error = predictions - y_train
    loss = np.mean(error ** 2)

    # Backpropagation
    d_predictions = (2 / len(X_train)) * error

    dW2 = a1.T @ d_predictions
    db2 = np.sum(d_predictions, axis=0, keepdims=True)

    da1 = d_predictions @ W2.T
    dz1 = da1 * (1 - a1 ** 2)

    dW1 = X_train.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")


# Evaluate final training predictions
train_hidden = np.tanh(X_train @ W1 + b1)
train_predictions = train_hidden @ W2 + b2

train_mse = np.mean((train_predictions - y_train) ** 2)

# Evaluate held-out test predictions
test_hidden = np.tanh(X_test @ W1 + b1)
test_predictions = test_hidden @ W2 + b2

test_mse = np.mean((test_predictions - y_test) ** 2)

print("\nFinal results:")
print(f"Training MSE: {train_mse:.6f}")
print(f"Test MSE:     {test_mse:.6f}")

print("\nFirst five test predictions:")

for actual, predicted in zip(
    y_test[:5].flatten(),
    test_predictions[:5].flatten(),
):
    print(
        f"Actual: {actual:.4f} | "
        f"Predicted: {predicted:.4f} | "
        f"Error: {abs(actual - predicted):.4f}"
    )



# Additional test metrics
absolute_errors = np.abs(test_predictions - y_test)

mae = np.mean(absolute_errors)
rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
max_error = np.max(absolute_errors)

worst_index = np.argmax(absolute_errors)

print("\nDetailed test evaluation:")
print(f"MAE:       {mae:.6f}")
print(f"RMSE:      {rmse:.6f}")
print(f"Max error: {max_error:.6f}")

print("\nWorst test example:")
print("Scaled inputs:", X_test[worst_index])
print("Actual sigma:", y_test[worst_index, 0])
print("Predicted sigma:", test_predictions[worst_index, 0])



# Convert normalized inputs back to physical values
original_inputs = (
    X_test[worst_index] * X_std + X_mean
)

worst_alpha = original_inputs[0, 0]
worst_time = original_inputs[0, 1]

print("\nWorst example physical parameters:")
print(f"Diffusion coefficient: {worst_alpha:.6f}")
print(f"Time: {worst_time:.6f}")



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Create a grid covering the training parameter ranges
alpha_values = np.linspace(0.01, 0.2, 80)
time_values = np.linspace(0.1, 5.0, 80)

alpha_grid, time_grid = np.meshgrid(
    alpha_values,
    time_values,
)

# Convert grid into model inputs
grid_inputs = np.column_stack((
    alpha_grid.ravel(),
    time_grid.ravel(),
))

# Apply the same scaling used during training
scaled_grid = (grid_inputs - X_mean) / X_std

# Neural network predictions
grid_hidden = np.tanh(scaled_grid @ W1 + b1)
grid_predictions = grid_hidden @ W2 + b2

# Analytical physics reference
grid_actual = np.sqrt(
    sigma0**2
    + 2 * grid_inputs[:, 0] * grid_inputs[:, 1]
)

# Absolute error at each parameter combination
grid_errors = np.abs(
    grid_predictions.flatten() - grid_actual
)

error_map = grid_errors.reshape(alpha_grid.shape)

plt.figure(figsize=(10, 6))

plt.pcolormesh(
    alpha_grid,
    time_grid,
    error_map,
    shading="auto",
    cmap="viridis",
)

plt.colorbar(label="Absolute prediction error")

plt.scatter(
    [worst_alpha],
    [worst_time],
    color="red",
    edgecolors="white",
    label="Worst held-out test example",
)

plt.xlabel("Diffusion coefficient (alpha)")
plt.ylabel("Elapsed time")
plt.title("Neural Surrogate — Prediction Error Map")
plt.legend()
plt.tight_layout()

plt.savefig("surrogate_error_map.png", dpi=150)
plt.close()

print("Saved surrogate_error_map.png")


# Verify reusable prediction function
sample_inputs = np.array([
    [0.05, 1.0],
    [0.10, 2.0],
    [0.15, 3.0],
])

sample_predictions = predict_surrogate(
    sample_inputs,
    W1,
    b1,
    W2,
    b2,
    X_mean,
    X_std,
)

print("\nReusable prediction function:")
for inputs, prediction in zip(
    sample_inputs,
    sample_predictions,
):
    print(
        f"alpha={inputs[0]:.2f}, "
        f"time={inputs[1]:.2f}, "
        f"prediction={prediction:.4f}"
    )