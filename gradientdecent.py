import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Define the function and its derivatives
def f(x, y, z):
    return x**2 + y**2 + z**2 - x - y + z

def fx(x, y, z):
    return 2*x - 1

def fy(x, y, z):
    return 2*y - 1

def fz(x, y, z):   
    return 2*z + 1


# Set the gradient descent parameters
eta = 0.001         # Learning rate
eps = 1e-6          # Minimum step size
max_cnt = 10000     # Max number of iterations
def gradient_decent(eta, eps, max_cnt):
    # Initialize the starting point
    x, y, z = 0.0, 0.0, 0.0

    # Initialize variables to store the change magnitude and count of iterations
    changes = []  # List to store changes for plotting
    cnt = 0

    # Gradient descent loop
    for cnt in range(max_cnt):
        x_change = -eta * fx(x, y, z)
        y_change = -eta * fy(x, y, z)
        z_change = -eta * fz(x, y, z)
        
        # Update the parameters
        x += x_change
        y += y_change
        z += z_change
        
        # Calculate and store the magnitude of the change
        change = norm(np.array([x_change, y_change, z_change]))
        changes.append(change)
        
        # Increment the count

        if change < eps:
            print("x: {:.4f}, y: {:.4f}, z: {:.4f}".format(x, y, z))
            return x, y, z, f(x, y, z), cnt+1, changes, True  # Converged

    print("x: {:.4f}, y: {:.4f}, z: {:.4f}".format(x, y, z))
    return x, y, z, f(x, y, z), cnt+1, changes, False  # Not converged


        
etas = np.linspace(0.001, 0.1, 100)
max_eta = 0
for eta in etas:
    _, _, _, min_value, iterations, changes, converged = gradient_decent(eta, eps, max_cnt)
    if converged:
        max_eta = eta
        print(f"Converged for eta = {eta:.4f} after {iterations} iterations with min value = {min_value:.4f}")
    else:
        print(f"Failed to converge for eta = {eta:.4f}")
        break  # Stop if it doesn't converge for this eta

print(f"Largest eta for convergence: {max_eta}")

# Plot changes for the largest eta
_, _, _, _, _, changes, _ = gradient_decent(max_eta, eps, max_cnt)
plt.plot(changes)
plt.xlabel('Iteration')
plt.ylabel('Change in value')
plt.title(f'Convergence for eta = {max_eta}')
plt.grid(True)
plt.show()
