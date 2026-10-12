import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Make random initialization reproducible
rng = np.random.default_rng(42)

# Network architecture
input_size = 1
hidden_size = 16
output_size = 1

# Initialize weights and biases
W1 = rng.normal(0, 0.5, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = rng.normal(0, 0.5, (hidden_size, output_size))
b2 = np.zeros((1, output_size))

# One input example
x = np.array([[2.0]])

# Forward pass: input -> hidden -> output
z1 = x @ W1 + b1
a1 = np.tanh(z1)

z2 = a1 @ W2 + b2
prediction = z2

print("Input shape:", x.shape)
print("Hidden layer shape:", a1.shape)
print("Output shape:", prediction.shape)
print("Prediction:", prediction[0, 0])
print("Actual sin(x):", np.sin(x[0, 0]))



# Generate the complete dataset
x_all = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
y_all = np.sin(x_all)

# Shuffle indices reproducibly
split_rng = np.random.default_rng(123)
indices = split_rng.permutation(len(x_all))

# 80% for training, 20% for testing
split = int(0.8 * len(x_all))

train_indices = indices[:split]
test_indices = indices[split:]

x_train = x_all[train_indices]
y_train = y_all[train_indices]

x_test = x_all[test_indices]
y_test = y_all[test_indices]

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))


# Training settings
learning_rate = 0.01
epochs = 20000

for epoch in range(epochs):
    # Forward pass
    z1 = x_train @ W1 + b1
    a1 = np.tanh(z1)

    prediction = a1 @ W2 + b2

    # Mean squared error
    error = prediction - y_train
    loss = np.mean(error ** 2)

    # Backpropagation: output layer
    d_prediction = (2 / len(x_train)) * error

    dW2 = a1.T @ d_prediction
    db2 = np.sum(d_prediction, axis=0, keepdims=True)

    # Backpropagation: hidden layer
    da1 = d_prediction @ W2.T
    dz1 = da1 * (1 - a1 ** 2)

    dW1 = x_train.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")

# Calculate final loss using updated weights
final_hidden = np.tanh(x_train @ W1 + b1)
final_predictions = final_hidden @ W2 + b2
final_loss = np.mean((final_predictions - y_train) ** 2)

print(f"Final training loss: {final_loss:.6f}")

# Evaluate the trained network
test_x = np.array([[2.0]])
test_hidden = np.tanh(test_x @ W1 + b1)
test_prediction = test_hidden @ W2 + b2

print("\nAfter training:")
print("Prediction:", test_prediction[0, 0])
print("Actual:", np.sin(2.0))


# Evaluate on unseen test examples
test_hidden = np.tanh(x_test @ W1 + b1)
test_predictions = test_hidden @ W2 + b2

test_loss = np.mean(
    (test_predictions - y_test) ** 2
)

print(f"Test loss: {test_loss:.6f}")
print(f"Training loss: {final_loss:.6f}")
print(f"Generalization gap: {test_loss - final_loss:.6f}")


# Extrapolation experiment
x_outside = np.array([
    [-5.0],
    [-4.0],
    [4.0],
    [5.0],
])

# Use the trained network
outside_hidden = np.tanh(x_outside @ W1 + b1)
outside_predictions = outside_hidden @ W2 + b2

outside_actual = np.sin(x_outside)

print("\nExtrapolation experiment:")
print("Input | Actual | Prediction | Absolute error")

for x_value, actual, predicted in zip(
    x_outside.flatten(),
    outside_actual.flatten(),
    outside_predictions.flatten(),
):
    absolute_error = abs(actual - predicted)

    print(
        f"{x_value:5.1f} | "
        f"{actual:7.4f} | "
        f"{predicted:10.4f} | "
        f"{absolute_error:13.4f}"
    )


# Create a sorted grid for visualization
x_plot = np.linspace(-8, 8, 400).reshape(-1, 1)

# Calculate true values
y_plot = np.sin(x_plot)

# Predict using trained weights
hidden_plot = np.tanh(x_plot @ W1 + b1)
predictions_plot = hidden_plot @ W2 + b2

plt.figure(figsize=(10, 6))

plt.plot(
    x_plot,
    y_plot,
    label="Actual sin(x)",
    linewidth=2,
)

plt.plot(
    x_plot,
    predictions_plot,
    label="Neural network prediction",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.title("Neural Network Generalization")
plt.legend()
plt.grid(True)

plt.savefig("generalization.png")
plt.close()

print("Saved generalization.png")