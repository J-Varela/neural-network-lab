# Neural Network Lab

A hands-on learning project focused on understanding how neural networks work, from individual neurons and backpropagation to scientific machine learning and surrogate modeling.

The goal is to learn by implementing models, running experiments, evaluating predictions, and understanding why they succeed or fail.

## What I've Built

### Neural Network Fundamentals
- Implemented an artificial neuron with weights, bias, and `tanh` activation.
- Built a fully connected neural network using NumPy.
- Implemented forward propagation, backpropagation, and gradient descent.
- Trained neural networks to approximate the sine function.
- Compared different hidden-layer sizes and training durations.

### Generalization Experiments
- Split datasets into training and test examples.
- Evaluated interpolation and extrapolation.
- Visualized how predictions behave outside the training domain.

### Diffusion Surrogate Modeling
- Generated synthetic data using an analytical heat-diffusion relationship.
- Trained a neural network to predict diffusion width from diffusivity and elapsed time.
- Evaluated predictions using MSE, MAE, RMSE, and maximum absolute error.
- Visualized prediction errors across the physical parameter space.

## Current Results

| Metric | Diffusion Surrogate |
|---|---:|
| Training examples | 400 |
| Test examples | 100 |
| Training MSE | 0.000415 |
| Test MSE | 0.000558 |
| Test MAE | 0.018272 |
| Test RMSE | 0.023612 |
| Maximum test error | 0.093238 |

These results come from a controlled experiment using an analytical diffusion model, not a full numerical PDE simulation.

## Tech Stack

- Python
- NumPy
- Matplotlib
- uv

## Getting Started

Clone the repository and install dependencies:

```bash
uv sync
```

Run the neural network experiments:

```bash
uv run python neuron.py
uv run python network.py
uv run python generalization.py
uv run python diffusion_surrogate.py
```

Generated plots are saved locally as PNG files.

## Next Experiments

- Compare random sampling with error-guided experiment selection.
- Measure how additional training data affects surrogate accuracy.
- Investigate model architecture and optimization strategies.
- Connect neural surrogate modeling to a numerical diffusion simulator.

## Project Goal

Develop a practical understanding of machine learning and scientific computing by building models from scratch, testing their limitations, and applying them to physical systems.

This repository is a learning project, with experiments added as the work progresses.