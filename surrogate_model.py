
import numpy as np


def initialize_model(input_size=2, hidden_size=16, seed=42):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(0, 0.3, (input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))

    W2 = rng.normal(0, 0.3, (hidden_size, 1))
    b2 = np.zeros((1, 1))

    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2):
    hidden = np.tanh(X @ W1 + b1)
    predictions = hidden @ W2 + b2

    return hidden, predictions


def predict(X, W1, b1, W2, b2, X_mean, X_std):
    X_scaled = (X - X_mean) / X_std

    _, predictions = forward(
        X_scaled, W1, b1, W2, b2
    )

    return predictions.reshape(-1)



def train_model(
    X_train,
    y_train,
    hidden_size=16,
    learning_rate=0.01,
    epochs=10000,
    seed=42,
):
    # Normalize using training data only
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True)

    if np.any(X_std == 0):
        raise ValueError("Input features must have nonzero variance")

    X_scaled = (X_train - X_mean) / X_std

    # Initialize network parameters
    W1, b1, W2, b2 = initialize_model(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        seed=seed,
    )

    y_train = np.asarray(y_train).reshape(-1, 1)

    for epoch in range(epochs):
        # Forward pass
        hidden, predictions = forward(
            X_scaled, W1, b1, W2, b2
        )

        error = predictions - y_train
        loss = np.mean(error ** 2)

        # Output-layer gradients
        d_predictions = (2 / len(X_scaled)) * error

        dW2 = hidden.T @ d_predictions
        db2 = np.sum(
            d_predictions, axis=0, keepdims=True
        )

        # Hidden-layer gradients
        d_hidden = d_predictions @ W2.T
        dz1 = d_hidden * (1 - hidden ** 2)

        dW1 = X_scaled.T @ dz1
        db1 = np.sum(
            dz1, axis=0, keepdims=True
        )

        # Gradient descent
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Measure loss after the final weight update
    _, final_predictions = forward(
        X_scaled, W1, b1, W2, b2
    )

    final_loss = np.mean(
        (final_predictions - y_train) ** 2
    )

    return (
        W1, b1, W2, b2,
        X_mean, X_std,
        float(final_loss),
    )
