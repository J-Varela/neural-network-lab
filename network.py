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


# Training dataset
x_train = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
y_train = np.sin(x_train)

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


# Generate predictions using trained weights
hidden = np.tanh(x_train @ W1 + b1)
predictions = hidden @ W2 + b2

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))

plt.plot(
    x_train,
    y_train,
    label="Actual sin(x)",
    linewidth=2,
)

plt.plot(
    x_train,
    predictions,
    label="Neural network prediction",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.title("Neural Network Learning sin(x)")
plt.legend()
plt.grid(True)

plt.savefig("trained_network.png")
plt.close()

print("Saved trained_network.png")