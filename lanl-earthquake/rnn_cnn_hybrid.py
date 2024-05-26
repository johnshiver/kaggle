import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Example of data normalization and feature extraction
def preprocess_data(data, segment_length=150000):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    segments = []
    for start in range(0, len(data) - segment_length, segment_length):
        segment = data_scaled[start : start + segment_length]
        segments.append(segment)
    return np.array(segments)


# Convert to PyTorch Dataset and DataLoader
class EarthquakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_channels=16, lstm_hidden_size=64, lstm_layers=2):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Prepare for LSTM (batch, seq, feature)
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last LSTM cell
        return x


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


# Load and preprocess the training data
train_data = pd.read_csv("~/datasets/LANL-Earthquake-Prediction/train.csv")
acoustic_data = train_data["acoustic_data"].values.reshape(-1, 1)
time_to_failure = train_data["time_to_failure"].values[
    ::150000
]  # Assuming each segment ends at every 150000th sample

# Preprocess data
X_train = preprocess_data(acoustic_data)
y_train = time_to_failure


train_dataset = EarthquakeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, criterion, and optimizer
model = CNNLSTM(input_dim=X_train.shape[2])
criterion = nn.L1Loss()  # MAE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=10)
