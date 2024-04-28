from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

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

def evaluate_model(model, X_test, y_test, filenames_test):
    """
    Evaluates the trained model using the test set with filenames, providing a simplified classification report.
    Returns the evaluation results as a string.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)  # Returns results for each class
    recall = recall_score(y_test, predictions, average=None)  # Returns results for each class

    results = f"Accuracy: {accuracy:.2f}\n"
    results += f"Precision per class: {precision}\n"
    results += f"Recall per class: {recall}\n\n"

    for filename, actual, predicted in zip(filenames_test, y_test, predictions):
        actual_label = 'Female' if actual == 1 else 'Male'
        predicted_label = 'Female' if predicted == 1 else 'Male'
        results += f"File: {filename}, Actual: {actual_label}, Predicted: {predicted_label}\n"

    return results
def save_model(model, filename):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Loads a trained model from disk.
    """
    return joblib.load(filename)
