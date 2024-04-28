import numpy as np

# Constants
eta = 0.7763  # Learning rate
seedy = 42  # Seed for random number generator
NUM_HIDDEN = 4  # Number of hidden nodes
NUM_EPOCHS = 6000  # Number of training epochs

def sigmoid(x):

    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):

    return x * (1.0 - x)

class ToyNN:
    def __init__(self, x, y):

        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], NUM_HIDDEN)
        self.weights2 = np.random.rand(NUM_HIDDEN, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):

        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):

        diff_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        diff_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += eta * diff_weights1
        self.weights2 += eta * diff_weights2

    def predict(self, input1):

        self.layer1 = sigmoid(np.dot(input1, self.weights1))
        return sigmoid(np.dot(self.layer1, self.weights2))
    


    def test_all_patterns(self):
        """Test all patterns and calculate average error, print detailed results for each."""
        self.feedforward()  # This will set the self.output to the latest network output for all inputs
        EPSILON = 1e-8
        errors = np.abs(self.y - self.output) / (np.abs(self.y) + EPSILON)
        errors = np.clip(errors, 0, 1)  # Clip errors to avoid unrealistic high values
        average_error = np.mean(errors)
        
        for i in range(len(self.input)):
            actual_count = self.y[i, 0] * 4  # Scale back the target count to the actual range
            predicted_count = self.output[i, 0] * 4
            print(f"Input: {self.input[i]} Actual: {actual_count:.0f}, Predicted: {predicted_count:.2f}, Error: {errors[i, 0] * 100:.1f}%")
        
        print(f"Average Relative Error for all patterns: {average_error * 100:.2f}%")
        return average_error
    
if __name__ == "__main__":
    np.random.seed(seedy)
    X = np.array([[int(x) for x in "{:04b}".format(i)] for i in range(16)], dtype=np.float32)
    y = np.array([[bin(i).count("1")] for i in range(16)], dtype=np.float32) / 4

    # Train with all patterns
    nn = ToyNN(X, y)
    for epoch in range(NUM_EPOCHS):
        nn.feedforward()
        nn.backprop()

    print("\n--- After initial training ---")
    nn.test_all_patterns()  # Calculate and print error after initial training with detailed results

    # Retrain excluding one pattern
    OMIT_PATTERN = 6  # Index of the pattern '0110'
    train_indices = list(set(range(16)) - {OMIT_PATTERN})
    nn = ToyNN(X[train_indices], y[train_indices])
    for epoch in range(NUM_EPOCHS):
        nn.feedforward()
        nn.backprop()

    # Print setup parameters
    print(f"Learning Rate: {eta}")
    print(f"Seed for Random Number Generator: {seedy}")
    print(f"Number of Hidden Nodes: {NUM_HIDDEN}")
    print(f"Number of Training Epochs: {NUM_EPOCHS}")

    print(f"\n--- After retraining without pattern {X[OMIT_PATTERN]} ---")
    nn.test_all_patterns()  # Calculate and print error after retraining with detailed results

    # Test on the omitted pattern
    omitted_output = nn.predict(X[OMIT_PATTERN].reshape(1, -1)) * 4
    print(f"\nOmitted Pattern '{X[OMIT_PATTERN]}' Test Output: {omitted_output.flatten()[0]:.2f} Actual Value: 2, Error: {np.abs(2 - omitted_output[0, 0]) * 100:.1f}%\n")

    