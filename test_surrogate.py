
import numpy as np

from surrogate_model import train_model, predict

# Generate a simple learning problem
rng = np.random.default_rng(123)

X = rng.uniform(0, 1, (100, 2))
y = X[:, 0] + 2 * X[:, 1]

# Train the network
W1, b1, W2, b2, X_mean, X_std, loss = train_model(
    X,
    y,
    epochs=5000,
)

# Generate predictions
predictions = predict(
    X, W1, b1, W2, b2, X_mean, X_std
)

mae = np.mean(np.abs(predictions - y))

print("Training loss:", loss)
print("Mean absolute error:", mae)
print("First prediction:", predictions[0])
print("First actual:", y[0])
