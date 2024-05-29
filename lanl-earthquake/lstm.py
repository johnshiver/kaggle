import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)


# Downsample the data by taking the mean of every nth values
def downsample_data(df, factor):
    df_acoustic = (
        df["acoustic_data"]
        .groupby(np.arange(len(df)) // factor)
        .mean()
        .reset_index(drop=True)
    )
    df_time = (
        df["time_to_failure"]
        .groupby(np.arange(len(df)) // factor)
        .max()
        .reset_index(drop=True)
    )
    return pd.DataFrame({"acoustic_data": df_acoustic, "time_to_failure": df_time})


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
        end_idx = start_idx + self.sequence_length
        x = self.data["acoustic_data"].iloc[start_idx:end_idx].values.astype("float32")
        y = self.data["time_to_failure"].iloc[end_idx - 1].astype("float32")
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


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    early_stopping_patience=5,
):
    best_val_loss = float("inf")
    early_stopping_counter = 0
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_time = time.time()

        for batch_idx, (X_batch, y_batch) in enumerate(train_dataloader):
            X_batch = X_batch.unsqueeze(
                -1
            ).cuda()  # Add channel dimension and move to GPU
            y_batch = y_batch.unsqueeze(-1).cuda()  # Move to GPU

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mixed precision training
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss = evaluate_model(model, val_dataloader, criterion)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Time: {time.time() - start_time:.2f}s, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break


def evaluate_model(model, dataloader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.unsqueeze(-1).cuda()
            y_batch = y_batch.unsqueeze(-1).cuda()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(dataloader)
    return val_loss


# Load and process the data
file_path = "lanl-earthquake/train.csv"
data = load_data(file_path)
downsample_factor = 10  # Adjust this factor as needed
downsampled_data = downsample_data(data, downsample_factor)

# Split data into training and validation sets
train_data = downsampled_data.sample(frac=0.8, random_state=42)
val_data = downsampled_data.drop(train_data.index)

# Define sequence length and stride
sequence_length = 150000
stride = 50000

# Create datasets and dataloaders
train_dataset = EarthquakeDataset(train_data, sequence_length, stride)
val_dataset = EarthquakeDataset(val_data, sequence_length, stride)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=12)

# Initialize model, criterion, optimizer, and scheduler
model = LSTMModel().cuda()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)

# Train the model
train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
)
