import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
        .min()
        .reset_index(drop=True)
    )
    return pd.DataFrame({"acoustic_data": df_acoustic, "time_to_failure": df_time})


# Feature Engineering
def create_features(df):
    df["moving_avg"] = df["acoustic_data"].rolling(window=50).mean()
    df["moving_std"] = df["acoustic_data"].rolling(window=50).std()
    df["exp_moving_avg"] = df["acoustic_data"].ewm(span=50).mean()
    df = df.fillna(0)
    return df


# Function to create additional features from the seismic data
def create_additional_features(df):
    df["acoustic_data_mean"] = df["acoustic_data"].mean()
    df["acoustic_data_std"] = df["acoustic_data"].std()
    df["acoustic_data_max"] = df["acoustic_data"].max()
    df["acoustic_data_min"] = df["acoustic_data"].min()
    df["acoustic_data_range"] = df["acoustic_data_max"] - df["acoustic_data_min"]
    return df


# Load and process the data
file_path = "lanl-earthquake/train.csv"
data = load_data(file_path)

data = downsample_data(data, 5)
data = create_features(data)

# Get earthquake indices
def get_earthquake_indices(df, threshold=0.01):
    earthquake_indices = []
    for i, ttf in enumerate(df["time_to_failure"]):
        if ttf < threshold:
            earthquake_indices.append(i)
    return earthquake_indices


earthquake_indices = get_earthquake_indices(data)
data = create_additional_features(data)

# Prepare the dataset for XGBoost
X = data.drop(columns=["time_to_failure"])
y = data["time_to_failure"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset into DMatrix format required by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define the parameters for XGBoost
params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "gpu_hist",  # Use GPU for training
}

# Train the XGBoost model
num_boost_round = 1000
early_stopping_rounds = 50
evals = [(dtrain, "train"), (dval, "eval")]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=10,
)

# Make predictions on the validation set
y_pred = model.predict(dval)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {mae:.4f}")
