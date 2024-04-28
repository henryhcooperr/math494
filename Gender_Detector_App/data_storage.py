import os
import numpy as np
from model import train_decision_tree, evaluate_model, save_model
import extract_audio_features
from function_tracker import count_function_calls

@count_function_calls
def save_features_to_file(features, labels, filenames, file_path):
    """
    Save features, labels, and filenames to a file, ensuring labels are saved as integers.
    """
    with open(file_path, 'w') as f:
        for feature, label, filename in zip(features, labels, filenames):
            # Ensure label is stored as an integer
            f.write(f"{filename},{int(label)},{' '.join(map(str, feature))}\n")


@count_function_calls
def load_features_from_file(feature_file):
    filenames = []
    labels = []
    features = []
    with open(feature_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Assuming comma is the delimiter
            if len(parts) > 2:
                filenames.append(parts[0])
                labels.append(int(parts[1]))

                try:
                    feature_list = parts[2].split()  # Splitting the feature string into individual numbers
                    features.append([float(x) for x in feature_list])
                except ValueError as e:
                    print("Error converting string to float:", e)
                    print("Offending data:", parts[2])
    print(labels)
    return filenames, labels, np.array(features)

