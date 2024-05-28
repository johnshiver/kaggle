import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Custom Dataset class
class EarthquakeDataset(Dataset):
    def __init__(self, data, sequence_length=150000):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        x = self.data["acoustic_data"].iloc[start_idx:start_idx + self.sequence_length].values.astype('float32')
        return torch.tensor(x)

    #def __len__(self):
    #    return len(self.data) - self.sequence_length

    #def __getitem__(self, idx):
    #    X = self.data["acoustic_data"].values[idx : idx + self.sequence_length]
    #    y = self.data["time_to_failure"].values[idx + self.sequence_length]
    #    return torch.tensor(X, dtype=torch.float32), torch.tensor(
    #        y, dtype=torch.float32
    #    )

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(
            x.device
        )
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(
            x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

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
    predictions = {}
    for file_name in os.listdir(test_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(test_dir, file_name)
            data = load_data(file_path)
            prediction = predict_time_to_failure(model, data, sequence_length)
            print(f"Predictions for {file_name}: {prediction}")
            predictions[file_name] = prediction
    return predictions

# Example usage:
#test_dir = "/media/johnshiver/hdd-fast/lanl-earthquake/test"
test_dir = "lanl-earthquake/test"
predictions = predict_from_test_directory(test_dir)

# Print the predictions
for file_name, prediction in predictions.items():
    print(f"Predictions for {file_name}: {prediction}")
