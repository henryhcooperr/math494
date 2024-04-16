import numpy as np

A = np.array([[0.70, 0.05, 0.10],
              [0.10, 0.55, 0.15],
              [0.20, 0.40, 0.75]])

X0 = np.array([40, 40, 40]) # Initial distribution

# Calculating after 5 years
A_5 = np.linalg.matrix_power(A, 5)
X_5 = A_5.dot(X0)

# Calculating after 10 years
A_10 = np.linalg.matrix_power(A, 10)
X_10 = A_10.dot(X0)

print("After 5 years:", X_5)
print("After 10 years:", X_10)