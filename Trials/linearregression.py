import numpy as np
import matplotlib.pyplot as plt

# Last digit of ID
a = 4

# User input for 'b'
b = float(input("Please enter a value for b: ")) # b is the slope of the exponential function

# b is the slope of the exponential function

# Generating the dataset
n = 100  # Number of observations
x = np.linspace(0, 5, n)
y = a * np.exp(b * x)
noise = np.random.normal(0, 10, size=y.shape)  # Adjust the noise level as needed
y_noisy = y + noise

# Ensure all noisy y values are positive
y_noisy = np.where(y_noisy <= 0, y * np.random.uniform(0.9, 1.1, size=y.shape), y_noisy)

# Linearizing the dataset
y_noisy_log = np.log(y_noisy)

# Manual calculation of linear regression parameters using the provided formulas
sum_xi_yi = np.sum(x * y_noisy_log)
sum_xi = np.sum(x)
sum_yi = np.sum(y_noisy_log)
sum_xi2 = np.sum(x**2)

# Calculate 'a' (slope) using the provided formulas
a_estimated = (n * sum_xi_yi - sum_xi * sum_yi) / (n * sum_xi2 - sum_xi**2)

# Calculate 'b' (intercept)
b_estimated = (sum_yi - a_estimated * sum_xi) / n

# Plotting the noisy dataset
plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy, label='Noisy Data')
plt.plot(x, y, 'r-', label='Original Model')
plt.title('Noisy Dataset with Exponential Trend')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Printing estimated values
print(f"Original a: {a}, Estimated a: {np.exp(b_estimated):.2f}")
print(f"Input b: {b}, Estimated b: {a_estimated:.2f}")

# Plotting the linearized dataset and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy_log, label='Linearized Noisy Data')
plt.plot(x, b_estimated + a_estimated * x, 'r-', label='Manual Linear Regression Line')
plt.title('Linearized Dataset with Manual Linear Regression')
plt.xlabel('x')
plt.ylabel('log(y)')
plt.legend()
plt.show()
