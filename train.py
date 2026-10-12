import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Generate 200 examples between -pi and pi.
x = np.linspace(-np.pi, np.pi, 200)
y = np.sin(x)

# Visualize the examples.
plt.plot(x, y)
plt.xlabel("Input (x)")
plt.ylabel("Target (sin(x))")
plt.title("Our Training Data")
plt.grid(True)
plt.savefig("training_data.png")

print("Training examples:", len(x))
print("First input:", x[0])
print("First target:", y[0])
