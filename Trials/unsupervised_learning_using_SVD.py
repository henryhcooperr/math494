import numpy as np

M = 8
Matrix = np.zeros((M, 5))
Matrix[:, :2] = np.random.uniform(0.0, 10.0, size=(M, 2))
print("Matrix with col 1-2 filled: \n" + str(Matrix) + "\n")

A = 4 # Example constants
B = 5
Matrix[:, 2] = Matrix[:, 0] + Matrix[:, 1]
Matrix[:, 3] = A*Matrix[:, 0] + B*Matrix[:, 1]
Matrix[:, 4] = B*Matrix[:, 0] + A*Matrix[:, 1]

# Adding noise
STD_dev = 1.5
mean = 0.0
Matrix[:, 2:] += np.random.normal(mean, STD_dev, (M, 3))
print("Matrix with noise: \n" + str(Matrix) + "\n")


# Perform SVD 

AT_A = np.dot(Matrix.T, Matrix)
A_AT = np.dot(Matrix, Matrix.T) 

eigenvalues_AT_A, eigenvectors_AT_A = np.linalg.eig(AT_A)
eigenvalues_A_AT, eigenvectors_A_AT = np.linalg.eig(A_AT)


singular_values = np.sqrt(np.sort(eigenvalues_AT_A)[::-1])
S = np.zeros((M, 5))
S[:5, :5] = np.diag(singular_values)
print("Singular Values Matrix S:\n", S)

sorted_indices = np.argsort(eigenvalues_A_AT)[::-1]
U = eigenvectors_A_AT[:, sorted_indices]


print("Left Singular Vector Matrix U:")
print(U)


V = eigenvectors_AT_A[:, np.argsort(eigenvalues_AT_A)[::-1]]

Vt = V.T

print("Right Singular Vectors Matrix V^T:\n", Vt)


Matrix_reconstructed = np.dot(U, np.dot(S, Vt))
print("Reconstructed Matrix:\n", Matrix_reconstructed)




k = int(input("Enter the value of k (0, 1, 2, or 3): "))

S_reduced = S[:5-k, :5-k]
print("Reduced Singular Values Matrix S:\n", S_reduced)

U_reduced = U[:, :5-k]
Vt_reduced = Vt[:5-k, :]
Matrix_reconstructed_reduced = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))

average_relative_error = np.mean(np.abs(Matrix - Matrix_reconstructed_reduced) / np.abs(Matrix))
print("Average Relative Error: ", average_relative_error)
