# raw or sampled data
import numpy as np
import matplotlib.pyplot as plt


x_axis = [1, 2, 3, 4, 5]
y_axis = [10, 20, 25, 30, 40]

# Correct usage: pass x and y as positional arguments (or use the Axes API)
plt.plot(x_axis, y_axis, marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sample plot")
plt.grid(True)
plt.show()