import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Function to downsample data
def downsample_data(file_path, output_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Standardize the acoustic data
    scaler = StandardScaler()
    acoustic_data_scaled = scaler.fit_transform(
        data["acoustic_data"].values.reshape(-1, 1)
    ).flatten()

    # Find boundaries where time_to_failure changes
    boundaries = acoustic_data_scaled["time_to_failure"].diff().fillna(0).ne(0).cumsum()

    # Group by boundaries and calculate the mean and gradient of acoustic_data
    grouped = acoustic_data_scaled.groupby(boundaries).agg(
        {
            "acoustic_data": ["mean", lambda x: np.gradient(x).mean()],
            "time_to_failure": "first",
        }
    )
    grouped.columns = ["acoustic_mean", "acoustic_gradient", "time_to_failure"]

    # Save the downsampled data
    grouped.reset_index(drop=True).to_csv(output_path, index=False)


# Usage
input_file_path = "lanl-earthquake/train.csv"
output_file_path = "lanl-earthquake/downsampled_train.csv"
downsample_data(input_file_path, output_file_path)
