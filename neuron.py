import numpy as np

# One input 
x = 2.0

# Parameters of our neuron
weight = 0.5
bias = 0.1

# Weighted sum
z = weight * x + bias

# Activation function
output = np.tanh(z)

print("Input:", x)
print("Weight:", weight)
print("Bias:", bias)
print("Weighted sum:", z)
print("Neuron output:", output)