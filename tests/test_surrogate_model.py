
import numpy as np

from surrogate_model import (
    initialize_model,
    forward,
    predict,
    train_model,
)


def test_initialize_model_shapes():
    W1, b1, W2, b2 = initialize_model()

    assert W1.shape == (2, 16)
    assert b1.shape == (1, 16)
    assert W2.shape == (16, 1)
    assert b2.shape == (1, 1)


def test_forward_output_shape():
    W1, b1, W2, b2 = initialize_model()

    X = np.ones((5, 2))

    hidden, predictions = forward(X, W1, b1, W2, b2)

    assert hidden.shape == (5, 16)
    assert predictions.shape == (5, 1)


def test_training_reduces_loss():
    rng = np.random.default_rng(123)

    X = rng.uniform(0, 1, (100, 2))
    y = X[:, 0] + 2 * X[:, 1]

    W1, b1, W2, b2, X_mean, X_std, loss = train_model(
        X,
        y,
        epochs=5000,
    )

    predictions = predict(
        X, W1, b1, W2, b2, X_mean, X_std
    )

    mae = np.mean(np.abs(predictions - y))

    assert loss < 0.01
    assert mae < 0.1
