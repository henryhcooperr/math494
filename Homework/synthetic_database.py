import numpy as np
import matplotlib.pyplot as plt

# Assuming A and B from a hypothetical USD ID# for demonstration purposes

#student id = 009503654
A = 4  
B = 5


# Generate the dataset


# Columns 1 and 2: Uniform distributions
col1 = np.random.uniform(A, 2*A, 100)
col2 = np.random.uniform(B, 2*B, 100)

# Columns 3 to 5: Linear combinations with Gaussian noise

std_dev3 = 0.1 * np.mean(col1 + col2)
std_dev4 = 0.5 * np.mean(col1 + 2*col2)
std_dev5 = np.mean(2*col1 + col2)

col3 = col1 + col2 + np.random.normal(0, std_dev3, 100) # 0.1 is standard deviation
col4 = col1 + 2*col2 + np.random.normal(0, std_dev4, 100) #0.5 is standard deviation
col5 = 2*col1 + col2 + np.random.normal(0, std_dev5, 100) # 1 is standard deviation

# Combine into a dataset
data = np.vstack([col1, col2, col3, col4, col5]).T

# Display the three-dimensional cloud of points
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 2], data[:, 3], data[:, 4])
ax.set_xlabel('Column 3')
ax.set_ylabel('Column 4')
ax.set_zlabel('Column 5')
plt.title('3D Cloud of Points')
plt.show()

# Compute and display mean and standard deviation of each column
means = np.mean(data, axis=0)
std_devs = np.std(data, axis=0)
print("Mean of each column:", means)
print("Standard deviation of each column:", std_devs)

# Center and standardize all data
data_centered_standardized = (data - means) / std_devs

# Compute and display the mean and standard deviation after centering and standardization
means_centered_standardized = np.mean(data_centered_standardized, axis=0)
std_devs_centered_standardized = np.std(data_centered_standardized, axis=0)
print("Mean of each column after centering and standardization:", means_centered_standardized)
print("Standard deviation of each column after centering and standardization:", std_devs_centered_standardized)
