import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)


# Downsample the data by taking the mean of every x rows
def downsample_data(df, factor):
    downsampled_df = df.groupby(df.index // factor).mean()
    return downsampled_df


# Custom Dataset class with overlapping windows
class EarthquakeDataset(Dataset):
    def __init__(self, data, sequence_length=150000, stride=50000):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride

        # Standardize the acoustic data
        scaler = StandardScaler()
        acoustic_data_scaled = scaler.fit_transform(
            data["acoustic_data"].values.reshape(-1, 1)
        ).flatten()

        # Update the data with the scaled values
        self.data = data.copy()
        self.data["acoustic_data"] = acoustic_data_scaled

    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.stride + 1

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = (
            self.data["acoustic_data"]
            .iloc[start_idx : start_idx + self.sequence_length]
            .values.astype("float32")
        )
        y = (
            self.data["time_to_failure"]
            .iloc[start_idx + self.sequence_length - 1]
            .astype("float32")
        )
        return x, y


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_layer_size=128,
        output_size=1,
        num_layers=3,
        dropout=0.2,
        bidirectional=False,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
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
file_path = "lanl-earthquake/train.csv"

data = load_data(file_path)

# Downsample the data
downsample_factor = 10  # Example: downsample by taking the mean of every 10 rows
downsampled_data = downsample_data(data, downsample_factor)

# Create the dataset and dataloader
sequence_length = 150000
stride = 50000
dataset = EarthquakeDataset(downsampled_data, sequence_length, stride)

# Debug: Print the length of the dataset
print(f"Dataset length: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=12)

# Debug: Print the number of batches per epoch
num_batches = len(dataloader)
print(f"Number of batches per epoch: {num_batches}")

# Initialize model
model = LSTMModel().cuda()

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.unsqueeze(-1).cuda()  # Add channel dimension and move to GPU
        y_batch = y_batch.unsqueeze(-1).cuda()  # Move to GPU

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if (
            batch_idx + 1
        ) % 100 == 0:  # Adjust the frequency of progress statements as needed
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

# Save the model
torch.save(model.state_dict(), "lstm_earthquake_model_downsampling_1.pth")
