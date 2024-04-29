import os
import numpy as np
from extract_audio_features import get_audio_files, process_files
from model import train_decision_tree, evaluate_model, save_model, load_model
from data_storage import save_features_to_file, load_features_from_file
from function_tracker import count_function_calls
from sklearn.model_selection import train_test_split
from record_audio import record_sample

"""
LABELS STORED AS INT
FEATURES STORED AS FLOATS

TO DO: FILE CREATION
- OWN AUDIO EVALUATION


"""


def print_data_balance(labels):

    male_count = sum(1 for label in labels if label == 'male')
    female_count = sum(1 for label in labels if label == 'female')

@count_function_calls
def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    feature_file = os.path.join(current_directory, "features.csv")

    if os.path.exists(feature_file):
        print("Features file found, loading data...")
        filenames, labels, features = load_features_from_file(feature_file)
        # Split data into train and test sets
        features_train, features_test, labels_train, labels_test, filenames_train, filenames_test = train_test_split(
            features, labels, filenames, test_size=0.2, random_state=42)
    else:
        print("No features file found. Exiting...")
        return

    while True:
        print("\nMenu:")
        print("1. Train decision tree model")
        print("2. Evaluate model")
        print("3. Record audio")
        print("4. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            if len(features_train) > 0 and len(labels_train) > 0:
                print("Training decision tree model...")
                model = train_decision_tree(features_train, labels_train, filenames_train)
                save_model(model, "decision_tree_model.pkl")
                print("Model trained and saved successfully.")
            else:
                print("Insufficient data for training. Please check your dataset.")
        elif choice == "2":
            if len(features_test) > 0 and len(labels_test) > 0:
                print("Evaluating model...")
                model = load_model("decision_tree_model.pkl")
                evaluate_model(model, features_test, labels_test, filenames_test)
            else:
                print("Insufficient test data available. Please check your dataset.")
        elif choice == "3":
            print("recording audio...")

            file_path = record_sample()
            model = load_model("decision_tree_model.pkl")
            

        elif choice == "4":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()