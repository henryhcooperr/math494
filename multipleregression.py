import numpy as np
from sklearn.linear_model import LinearRegression

# Set the random seed for reproducibility
np.random.seed(42)

# Generate the dataset
dataset = np.random.uniform(0, 1, size=(100, 4))

# Set the coefficients based on the last four digits of your USD ID
usd_id_last_four_digits = [3, 6, 5, 4]
coefficients = usd_id_last_four_digits

# Compute the linear combination of the first four columns
column5 = np.dot(dataset[:, :4], coefficients)

# Calculate the standard deviation for noise
average_coefficients = np.mean(usd_id_last_four_digits)
std_dev_noise = average_coefficients / 5
noise = np.random.normal(0, std_dev_noise, size=(100,))

# Add noise to the linear combination
column5_with_noise = column5 + noise

# Compute the design matrix A (including a column of ones for the intercept)
A = np.hstack((np.ones((dataset.shape[0], 1)), dataset[:, :4]))

# Compute the target vector B
B = column5_with_noise.reshape(-1, 1)

# Compute the transpose of A
A_transpose = A.T

# Compute A^TA
A_transpose_A = np.dot(A_transpose, A)

# Compute the inverse of A^TA
inverse_A_transpose_A = np.linalg.inv(A_transpose_A)

# Compute A^TB
A_transpose_B = np.dot(A_transpose, B)

# Solve for X using the inverse method
coefficients_from_inverse_method = np.dot(inverse_A_transpose_A, A_transpose_B)

# Print the regression coefficients obtained from the inverse method
print("Coefficients from inverse method:")
print(coefficients_from_inverse_method.flatten())

# Create an instance of LinearRegression
regression = LinearRegression()

# Fit the model to the dataset
regression.fit(dataset[:, :4], column5_with_noise)

# Print the regression coefficients from sklearn LinearRegression
print("Coefficients from sklearn LinearRegression:")
print(regression.coef_)

# Compare the coefficients with the actual pattern (usd_id_last_four_digits)
print("\nComparison with the actual pattern:")
print(f"Actual pattern: {usd_id_last_four_digits}")
print(f"Coefficients from inverse method: {coefficients_from_inverse_method[1:].flatten()}")
print(f"Coefficients from sklearn LinearRegression: {regression.coef_}")

# Calculate differences between methods and actual coefficients
difference_inverse_method = np.abs(usd_id_last_four_digits - coefficients_from_inverse_method[1:].flatten())
difference_sklearn = np.abs(usd_id_last_four_digits - regression.coef_)

# Print comparison
print("\nDifferences between actual coefficients and inverse method:", difference_inverse_method)
print("Differences between actual coefficients and sklearn LinearRegression:", difference_sklearn)

# Conclusion based on differences
print("\nConclusion:")
print("Both methods provide coefficients close to the actual pattern, with minor differences due to the added noise. This demonstrates the effectiveness of both the manual inverse method implementation and sklearn's LinearRegression in identifying the underlying pattern. The differences observed highlight the impact of noise on the precision of coefficient estimation.")
