import os
import numpy as np
from extract_audio_features import get_audio_files, process_files
from model import train_decision_tree, evaluate_model, save_model
from data_storage import save_features_to_file, load_features_from_file

def print_data_balance(labels):
    male_count = sum(1 for label in labels if label == 'male')
    female_count = sum(1 for label in labels if label == 'female')
    print(f"Male samples: {male_count}")
    print(f"Female samples: {female_count}")

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    male_audio_directory = os.path.join(current_directory, "male_mp3")
    female_audio_directory = os.path.join(current_directory, "female_mp3")
    feature_file = os.path.join(current_directory, "features.csv")

    testing_mode = input("Are you in testing mode? (y/n): ").lower() == 'y'
    file_limit = int(input("Enter the maximum number of files to process in testing mode: ")) if testing_mode else None

    
    if not os.path.exists(feature_file):



        # Extract features for both training and testing
        male_train_files, male_train_labels, male_test_files, male_test_labels = get_audio_files(male_audio_directory)
        female_train_files, female_train_labels, female_test_files, female_test_labels = get_audio_files(female_audio_directory)
            
        print("Extracting Male Audio Features for Training:")
        male_train_features, male_filenames, male_train_labels = process_files(male_train_files, male_train_labels, file_limit)
        print("Extracting Female Audio Features for Training:")
        female_train_features, female_filenames, female_train_labels = process_files(female_train_files, female_train_labels, file_limit)

        print("Extracting Male Audio Features for Testing:")
        male_test_features, male_test_filenames, male_test_labels = process_files(male_test_files, male_test_labels, file_limit)
        print("Extracting Female Audio Features for Testing:")
        female_test_features, female_test_filenames, female_test_labels = process_files(female_test_files, female_test_labels, file_limit)

        

        train_features = male_train_features + female_train_features
        labels = [1 if "female" in label else 0 for label in male_train_labels + female_train_labels]            
        filenames = male_filenames + female_filenames

        # Combine testing data
        test_features = male_test_features + female_test_features
        test_labels = [1 if "female" in label else 0 for label in male_test_labels + female_test_labels]
        test_filenames = male_test_filenames + female_test_filenames

            
        print_data_balance(labels)  # Print balance info for training data
        print("Feature extraction completed.")

        save_features_to_file(features, labels, filenames, feature_file)

        features, labels, filenames = load_features_from_file(feature_file)
    else:
        # Load features from file if it exists
        features, labels, filenames = load_features_from_file(feature_file)
        
        # Ensure the testing data is assigned
        male_test_files, male_test_labels, female_test_files, female_test_labels = get_audio_files(male_audio_directory, test_size=file_limit)
        male_test_features, male_test_filenames, male_test_labels = process_files(male_test_files, male_test_labels, file_limit)
        female_test_features, female_test_filenames, female_test_labels = process_files(female_test_files, female_test_labels, file_limit)
        
        test_features = male_test_features + female_test_features
        test_labels = [1 if "female" in label else 0 for label in male_test_labels + female_test_labels]
        test_filenames = male_test_filenames + female_test_filenames


    while True:
        print("\nMenu:")
        print("1. Extract features from audio files")
        print("2. Train decision tree model")
        print("3. Evaluate model")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("Training decision tree model...")
            model = train_decision_tree(np.array(features), np.array(labels), filenames)
            save_model(model, "decision_tree_model.pkl")
            print("Model trained and saved successfully.")
        elif choice == "2":
            if not features or not labels:
                print("No features or labels available. Please extract features first.")
            else:
                print("Training decision tree model...")
                model = train_decision_tree(np.array(features), np.array(labels), filenames)
                save_model(model, "decision_tree_model.pkl")
                print("Model trained and saved successfully.")

        elif choice == "3":
            if not test_features or not test_labels or not test_filenames:
                print("No test data available. Please extract features first.")
            else:
                print("Evaluating model on test data...")
                evaluate_model(model, np.array(test_features), np.array(test_labels), test_filenames)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()