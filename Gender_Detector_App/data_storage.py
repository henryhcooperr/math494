import os
import numpy as np
from model import train_decision_tree, evaluate_model, save_model
import extract_audio_features

def save_features_to_file(features, labels, filenames, file_path):
    """
    Save features, labels, and filenames to a file.
    """
    with open(file_path, 'w') as f:
        for feature, label, filename in zip(features, labels, filenames):
            f.write(f"{filename},{label},{' '.join(map(str, feature))}\n")


def load_features_from_file(file_path):
    """
    Load features, labels, and filenames from a file.
    """
    features = []
    labels = []
    filenames = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            filenames.append(parts[0])
            labels.append(parts[1])
            feature = list(map(float, parts[2].split()))
            features.append(feature)
    return features, labels, filenames



