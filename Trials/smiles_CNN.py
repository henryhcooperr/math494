import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

size = 2000
test_perc = 0.3

# Adjust the thresholds of "noise" to experiment with different levels of randomness
thresh1 = 0.5
thresh2 = 1.0

# Patterns for noughts and crosses:
nought = {6, 8, 10, 12, 14, 16, 18}  # Forming a simple circle
cross = {1, 5, 7, 9, 11, 13, 17, 21, 23}  # Forming an 'X'

# Generate the dataset for noughts and crosses:
X = np.zeros((size, 25))
y = np.zeros((size,))
for i in range(size):
    X[i,:] = np.random.uniform(0, 1, 25)
    ran = np.random.rand()
    for j in range(25):
        if ran < 0.5:
            if j in nought:
                X[i, j] = np.random.uniform(low=thresh1, high=thresh2)
        else:
            if j in cross:
                X[i, j] = np.random.uniform(low=thresh1, high=thresh2)
    y[i] = 0 if ran < 0.5 else 1

# Split data into train and test partitions:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=test_perc)

# Classifier parameters:
activation_type = 'relu'
solver_type = 'lbfgs'
max_iter_val = 500
num_nodes = 10

# Construct the MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(num_nodes,),
    activation=activation_type,
    max_iter=max_iter_val,
    alpha=1e-4,
    solver=solver_type,
    verbose=0,
    learning_rate_init=0.2,
)

# Fit the MLPClassifier catching a warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

# Obtain and display the confusion matrix
cm = confusion_matrix(y_test, mlp.predict(X_test))
print("Confusion matrix:\n")
print(cm)

# Print training and test set scores
print("\nTraining set score: %f" % mlp.score(X_train, y_train))
print("\nTest set score: %f" % mlp.score(X_test, y_test))

# Uncomment to visualize the first three test images

for k in range(3):
    fig = plt.figure()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    currentAxis = plt.gca()
    for i in range(5):
        for j in range(5):
            col = X_test[k, 5*j + i]
            currentAxis.add_patch(Rectangle((2*i, 2*j), 2, 2, facecolor=(col, col, col)))
    plt.show()

