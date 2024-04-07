import matplotlib.pyplot as plt
import numpy as np

# Define the range of x values
x = np.linspace(0, 1000000000)  # 400 points from -100 to 100

# Define the constants
mbr = 1
providedpool = 1
userfee = 0.005

# Compute y values based on the given formula
y = ((userfee * x) - (mbr - ((mbr/(providedpool + (userfee * x))) * (providedpool + (userfee * x)))))

#total pool
total_pool = providedpool + x * userfee

plt.plot(x, total_pool)

min_y = np.min(y)
min_x = x[np.argmin(y)]

max_y = np.max(y)
max_x = x[np.argmax(y)]

print(f"Minimum y value: {min_y}")
print(f"Corresponding x value: {min_x}")

print(f"Maximum y value: {max_y}")
print(f"Corresponding x value: {max_x}")

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = x * (0.005x - 0.01 + 0.00025(100 + x(0.005x)))')
plt.title('Visualization of the Formula')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
