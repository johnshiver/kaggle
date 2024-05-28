import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from lstm import EarthquakeDataset, LSTMModel


# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)


def predict_time_to_failure(model, data, sequence_length=150000):
    model.eval()
    dataset = EarthquakeDataset(data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for X_batch in dataloader:
            X_batch = X_batch.unsqueeze(-1).cuda()
            y_pred = model(X_batch)
            predictions.append(y_pred.item())

    return predictions


# Load the trained model
model = LSTMModel().cuda()
model.load_state_dict(torch.load("lstm_earthquake_model.pth"))


# Function to process all CSV files in the test directory and make predictions
def predict_from_test_directory(test_dir, sequence_length=150000):
    results = []
    for file_name in os.listdir(test_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(test_dir, file_name)
            data = load_data(file_path)
            prediction = predict_time_to_failure(model, data, sequence_length)
            for pred in prediction:
                results.append([file_name, pred])
    return results


# Example usage:
test_dir = "lanl-earthquake/test"
predictions = predict_from_test_directory(test_dir)

# Save the predictions to a CSV file
output_file = "predictions.csv"
predictions_df = pd.DataFrame(predictions, columns=["file_name", "prediction"])
predictions_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
