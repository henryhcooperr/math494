from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from function_tracker import count_function_calls

@count_function_calls
def train_decision_tree(X, y, filenames):
    """
    Trains a Decision Tree Classifier.
    Args:
        X (numpy.ndarray): Feature array.
        y (numpy.ndarray): Label array.
        filenames (list): List of filenames corresponding to X and y.
    Returns:
        DecisionTreeClassifier: Trained model.
    """
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, filenames_test) 
    return model

@count_function_calls
def train_random_forest(X, y, filenames):
    """
    Trains a Random Forest Classifier.
    Args:
        X (numpy.ndarray): Feature array.
        y (numpy.ndarray): Label array.
        filenames (list): List of filenames corresponding to X and y.
    Returns:
        RandomForestClassifier: Trained model.
    """
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, filenames_test)
    return model

from sklearn.metrics import accuracy_score, precision_score, recall_score

@count_function_calls
def evaluate_model(model, X_test, y_test, filenames_test):
    print("Filenames Test:" + str(filenames_test))
    
    # Ensure that y_test and predictions are of type int, as expected by precision and recall functions
    y_test = [1 if y == 'female' else 0 for y in y_test]
    predictions = model.predict(X_test)

    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')  # Changed to 'macro' for overall average
    recall = recall_score(y_test, predictions, average='macro')  # Changed to 'macro' for overall average

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision per class:", precision)
    print("Recall per class:", recall)

    # Print predictions against actuals for each file
    for filename, actual, predicted in zip(filenames_test, y_test, predictions):
        print(actual)
        print(predicted)
        actual_label = 'Female' if actual == 1 else 'Male'
        predicted_label = 'Female' if predicted == 1 else 'Male'
        print(f"File: {filename}, Actual: {actual_label}, Predicted: {predicted_label}")

@count_function_calls
def save_model(model, filename):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, filename)

@count_function_calls
def load_model(filename):
    """
    Loads a trained model from disk.
    """
    return joblib.load(filename)
