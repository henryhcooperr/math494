import numpy as np
import matplotlib.pyplot as plt


M = 1234  

# Number of times to repeat the process
N = 10000  

# Generate N sums of M pseudo-random numbers from U(0, 1)
uniform_sums = np.array([np.sum(np.random.uniform(0, 1, M)) for _ in range(N)])

# Parameters for the normal distribution
mu = M / 2 # Mean of the sum of M uniform random numbers
sigma = np.sqrt(M / 12) # Variance of the sum of M uniform random numbers

# Generate N pseudo-random numbers from the normal distribution
normal_prns = np.random.normal(mu, sigma, N)

# Plot the histograms
plt.hist(uniform_sums, bins=50, histtype='step', label='Sum of Uniform PRNs')
plt.hist(normal_prns, bins=50, histtype='step', label='Normal Distribution PRNs')

# Labeling the plot
plt.title('Comparison of Uniform Sums and Normal Distribution PRNs')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Show the plot
plt.show()
