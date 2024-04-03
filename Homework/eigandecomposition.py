import numpy as np

# Original non-square matrix
a = np.array([[1, 2, 3], [3, 1, 2]])

# Using string concatenation for printing arrays
print("Original 2x3 matrix:\n" + str(a))

# Define the two square matrices
a1 = np.array([[1, 2], [3, 1]])
a2 = np.array([[2, 1], [3, 2]])

# Print the two square matrices
print("The two square matrices are:")
print("a1:\n" + str(a1))
print("a2:\n" + str(a2))

def is_matrix_equal(matrix1, matrix2, tolerance=1e-10):
    if matrix1.shape != matrix2.shape:
        return False
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            if abs(matrix1[i, j] - matrix2[i, j]) > tolerance:
                return False
    return True

def perform_eigendecomposition(matrix):
    # Calculating eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Constructing the diagonal matrix D from eigenvalues
    D = np.diag(eigenvalues)
    
    # Inverse of the matrix of eigenvectors
    P_inv = np.linalg.inv(eigenvectors)
    
    # Reconstructing the original matrix
    new_matrix = np.matmul(np.matmul(eigenvectors, D), P_inv)

    print("Matrix P (Eigenvectors):")
    print(eigenvectors)
    print("Matrix D (Eigenvalues on the diagonal):")
    print(D)
    print("Matrix P^-1 (Inverse of P):")
    print(P_inv)
    print("Reconstructed Matrix (PDP^-1):")
    print(new_matrix)
    
    # Manually check if matrices are equal
    print("Is the reconstructed matrix equal to the original? ", is_matrix_equal(matrix, new_matrix))
    print("-------------------------------------------")

# Performing eigendecomposition on the matrices
print("Eigendecomposition for a1:")
perform_eigendecomposition(a1)

print("Eigendecomposition for a2:")
perform_eigendecomposition(a2)
