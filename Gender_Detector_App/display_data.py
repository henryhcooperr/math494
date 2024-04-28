import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import numpy as np
from extract_audio_features import get_audio_files, process_files
from model import train_decision_tree, evaluate_model, save_model, load_model

class AudioClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Classification Tool")
        master.geometry('400x250')  # Adjust the window size

        self.load_button = tk.Button(master, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.evaluate_button = tk.Button(master, text="Evaluate Model", command=self.evaluate)
        self.evaluate_button.pack(pady=10)

        self.exit_button = tk.Button(master, text="Exit", command=master.quit)
        self.exit_button.pack(pady=10)

        self.progress = ttk.Progressbar(master, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=20)

        self.result_text = tk.Text(master, height=10, width=50)
        self.result_text.pack(pady=10)


        self.model = None  # Model placeholder

        # Directories for audio files
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.male_audio_directory = os.path.join(current_directory, "male_mp3")
        self.female_audio_directory = os.path.join(current_directory, "female_mp3")

    def display_results(self, results):
        self.result_text.delete(1.0, tk.END)  # Clear previous results
        self.result_text.insert(tk.END, results)
    
    def load_data(self):
        try:
            # Ask if the program is in test mode
            test_mode = messagebox.askyesno("Test Mode", "Are you in testing mode?")

            # Ask for the maximum number of files to process if in test mode
            file_limit = None
            if test_mode:
                file_limit = simpledialog.askinteger("File Limit", "Enter the maximum number of files to process:")

            # Load features for both training and testing directly
            male_train_files, male_train_labels, male_test_files, male_test_labels = get_audio_files(self.male_audio_directory)
            female_train_files, female_train_labels, female_test_files, female_test_labels = get_audio_files(self.female_audio_directory)

            # Process files with updates on the progress bar
            self.progress['maximum'] = len(male_train_files) + len(female_train_files)
            self.progress['value'] = 0
            self.master.update()

            male_train_features, male_train_filenames, male_train_labels = process_files(male_train_files, male_train_labels, file_limit)
            female_train_features, female_train_filenames, female_train_labels = process_files(female_train_files, female_train_labels, file_limit)

            self.train_features = male_train_features + female_train_features
            self.train_labels = male_train_labels + female_train_labels
            self.train_filenames = male_train_filenames + female_train_filenames

            male_test_features, male_test_filenames, male_test_labels = process_files(male_test_files, male_test_labels, file_limit)
            female_test_features, female_test_filenames, female_test_labels = process_files(female_test_files, female_test_labels, file_limit)

            self.test_features = male_test_features + female_test_features
            self.test_labels = male_test_labels + female_test_labels
            self.test_filenames = male_test_filenames + female_test_filenames

            # Update progress bar after each file is processed
            for _ in male_train_features + female_train_features + male_test_features + female_test_features:
                self.progress['value'] += 1
                self.master.update_idletasks()

            messagebox.showinfo("Success", "Data loaded and processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.progress['value'] = 0  # Reset the progress bar

    def train_model(self):
        if hasattr(self, 'train_features') and hasattr(self, 'train_labels') and hasattr(self, 'train_filenames'):
            try:
                self.model = train_decision_tree(np.array(self.train_features), np.array(self.train_labels), self.train_filenames)
                save_model(self.model, 'decision_tree_model.pkl')
                messagebox.showinfo("Success", "Model trained and saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showinfo("Warning", "Load data before training the model.")

    def evaluate(self):
        if self.model is not None and hasattr(self, 'test_features') and hasattr(self, 'test_labels'):
            try:
                results = evaluate_model(self.model, np.array(self.test_features), np.array(self.test_labels), self.test_filenames)
                self.display_results(results)
                messagebox.showinfo("Evaluation", "Model evaluation completed.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showinfo("Warning", "Train the model and load test data before evaluation.")
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioClassifierApp(root)
    root.mainloop()