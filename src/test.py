import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate some random data
data = np.random.randn(1000)

# Calculate the kernel density estimation
kde = gaussian_kde(data)

# Define the range over which to evaluate the density
xmin, xmax = data.min(), data.max()
x = np.linspace(xmin, xmax, 1000)

# Evaluate the density on the defined range
density = kde(x)

# Create the density plot
fig, ax = plt.subplots()
ax.plot(x, density)
ax.fill_between(x, density, alpha=0.5)
ax.set_xlabel('Data')
ax.set_ylabel('Density')
ax.set_title('One-Dimensional Statistical Density Distribution')

plt.show()
