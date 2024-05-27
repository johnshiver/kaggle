import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)


# Custom Dataset class
class EarthquakeDataset(Dataset):
    def __init__(self, data, sequence_length=150000):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        X = self.data["acoustic_data"].values[idx : idx + self.sequence_length]
        y = self.data["time_to_failure"].values[idx + self.sequence_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


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


# Load and sample the data
file_path = "~/datasets/LANL-Earthquake-Prediction/train.csv"
data = load_data(file_path)
sample_size = 1000000  # Adjust this based on your memory and time constraints
data_sample = data.sample(n=sample_size, random_state=42)

# Prepare the data
sequence_length = 150000
dataset = EarthquakeDataset(data_sample, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Initialize the model, loss function, and optimizer
model = LSTMModel().cuda()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.unsqueeze(-1).cuda()  # Add channel dimension and move to GPU
        y_batch = y_batch.unsqueeze(-1).cuda()  # Move to GPU

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

# Save the model
torch.save(model.state_dict(), "lstm_earthquake_model.pth")
