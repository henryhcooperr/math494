import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Constants
M = 8
A = 4
B = 5
STD_dev = 1.5
mean = 0.0

# Generate the matrix
Matrix = np.zeros((M, 5))

Matrix[:, :2] = np.random.uniform(0.0, 10.0, size=(M, 2))
Matrix[:, 2] = Matrix[:, 0] + Matrix[:, 1]
Matrix[:, 3] = A * Matrix[:, 0] + B * Matrix[:, 1]
Matrix[:, 4] = B * Matrix[:, 0] + A * Matrix[:, 1]
Matrix[:, 2:] += np.random.normal(mean, STD_dev, (M, 3))
print("Noisy Dataset:\n", Matrix)

# Center the dataset
Matrix_centered = Matrix - np.mean(Matrix, axis=0)

# Perform PCA by hand
cov_matrix = np.cov(Matrix_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Compute the proportion of variance explained by each component
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
print("Proportion of Variance (by hand):\n", explained_variance)

# Perform PCA with sklearn
pca = PCA()
pca.fit(Matrix_centered)
print("Proportion of Variance (sklearn):\n", pca.explained_variance_ratio_)

# Perform SVD
U, S, Vt = np.linalg.svd(Matrix_centered)
singular_values = S**2 / (M - 1)
proportion_variance_svd = singular_values / np.sum(singular_values)
print("Proportion of Variance (SVD):\n", proportion_variance_svd)

#compare results
print("Comparison of Singular Values:\n")
print("Eigenvalues (by hand):", sorted_eigenvalues)
print("Singular Values Squared (PCA sklearn):", pca.explained_variance_)
print("Singular Values Squared (SVD):", singular_values)
