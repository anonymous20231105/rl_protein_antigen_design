import matplotlib.pyplot as plt
import numpy as np

# Data for the line graphs
x = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # X-axis values
y_rl_entire = [71.5, 70.2, 68.1, 67.9, 69.2, 71.8, 74.6, 79.2, 81.8]     # Y-axis values for Line 1
y2 = [72.5, 70.4, 66.2, 63.3, 61.4, 61.4, 60.5, 64.3, 68.1]      # Y-axis values for Line 2
y3 = [71.7, 62.8, 55.2, 48.1, 42.8, 36.4, 32.4, 29.6, 29.0]       # Y-axis values for Line 3
y4 = [74.6, 69.2, 63.9, 58.2, 55.2, 46.7, 43.0, 37.5, 34.3]       # Y-axis values for Line 4

# Create a new figure and axis
figsize = (16 * 0.35, 10 * 0.35)
fig, ax = plt.subplots(figsize=figsize)

# Set the x-axis and y-axis limits
# ax.set_xlim(0.1, 0.9)
# ax.set_ylim(0, 100)

# Calculate the standard deviation for each line
std1 = np.array([11.4, 11.0, 11.7, 8.9, 7.2, 4.6, 4.0, 2.9, 1.9])
std2 = np.array([12.3, 13.1, 16.0, 13.9, 13.8, 10.7, 11.9, 11.2, 7.3])
std3 = np.array([11.8, 11.6, 11.4, 11.5, 10.4, 7.9, 6.5, 6.8, 7.6])
std4 = np.array([12.4, 13.7, 14.4, 15.4, 15.7, 13.1, 11.1, 9.0, 8.4])

# Plot the line graphs
ax.plot(x, y_rl_entire, ".-", label='RL-Entire', alpha=0.8)
ax.plot(x, y2, ".-", label='RL-Specify', alpha=0.8)
ax.plot(x, y3, ".-", label='MC-Entire', alpha=0.8)
ax.plot(x, y4, ".-", label='MC-Specify', alpha=0.8)

# Shade the standard deviation
ax.fill_between(x, y_rl_entire - std1, y_rl_entire + std1, alpha=0.15)
ax.fill_between(x, y2 - std2, y2 + std2, alpha=0.15)
ax.fill_between(x, y3 - std3, y3 + std3, alpha=0.15)
ax.fill_between(x, y4 - std4, y4 + std4, alpha=0.15)

# Add a legend
ax.legend()

# Add labels and title
ax.set_xlabel('Prediction Rate (%)')
ax.set_ylabel('PLDDT (%)')
# ax.set_title('Line Graphs')

plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.15)

plt.savefig("../data/processed/fig_predictionPred_plddt.pdf")
print("Saved...")

# Show the plot
print("plt.show()...")
plt.show()
