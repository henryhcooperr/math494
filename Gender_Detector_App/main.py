import os 
import librosa
import numpy as np 
from sklearn.model_selection import train_test_split

import os

female_path = "/Users/henrycooper/Documents/GitHub/math494/Gender Detector Project/female_audio/0.m4a"
def load_audio_file(file_name):
    # Construct path relative to the current script
    current_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    file_path = os.path.join(current_dir, "female_audio", file_name)
    
    try:
        audio, sr = librosa.load(file_path, sr=1)  # Load with the original sample rate
        print("Audio loaded successfully")
        return audio, sr
    except Exception as e:
        print(f"Failed to load audio file: {e}")
        return None, None

file_name = "0.m4a"
audio, sr = load_audio_file(file_name)

def extract_features(audio, sr, n_mfcc=13): 
    if audio is None or len(audio) == 0: # Check if audio is empty
        raise ValueError("Empty Audio Data Provided")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc) # Mel-frequency cepstral coefficients
    return np.mean(mfcc, axis=1) # Return the average of the mfcc

