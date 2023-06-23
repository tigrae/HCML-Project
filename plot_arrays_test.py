import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.array([1, 2, 3, 4, 5])  # X-axis values
bar_width = 0.35  # Width of the bars
offset = bar_width / 2  # Offset to position bars side by side
y1 = np.array([10, 20, 15, 25, 30])  # Y-axis values for the first bar
y2 = np.array([5, 15, 10, 20, 25])  # Y-axis values for the second bar

# Plotting the bar chart
plt.bar(x - offset, y1, width=bar_width, label='Bar 1')
plt.bar(x + offset, y2, width=bar_width, label='Bar 2')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Bar Chart')

# Adding legend
plt.legend()

# Displaying the chart
plt.show()