from data import *
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Define your PyTorch model
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to create the model
def create_model(input_dim):
    return RegressionModel(input_dim)


# Load the dataset
dataset_df = load_dataset(TRAIN_FILE_PATH)

# Preprocess the features
X_preprocessed, y, preprocessor = preprocess_features(dataset_df)

# Convert y to a numpy array
y = y.to_numpy().astype(np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# Convert X_train and X_test to numpy arrays of type float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Initialize the Skorch regressor with the PyTorch model
net = NeuralNetRegressor(
    create_model,
    module__input_dim=X_train.shape[1],
    max_epochs=20,
    lr=0.1,
    optimizer=optim.Adam,
    criterion=nn.MSELoss,
    train_split=None,
)

# Define the parameter grid
params = {
    "lr": [0.01, 0.05, 0.1],
    "max_epochs": [20, 50, 100],
    "optimizer__weight_decay": [0, 0.01, 0.001],
    # Note: Adjust 'module__hidden_units' based on your model structure if necessary
    # 'module__hidden_units': [50, 100, 200],
}

# Perform Grid Search
gs = GridSearchCV(
    net, params, refit=True, cv=3, scoring="neg_mean_squared_error", verbose=2
)
gs.fit(X_train, y_train)

# Get the best parameters
print("Best parameters found: ", gs.best_params_)

# Use the best model
best_model = gs.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train).astype(np.float32)
y_pred_test = best_model.predict(X_test).astype(np.float32)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Load the test dataset
test_df = pd.read_csv(TEST_FILE_PATH)
ids = test_df["Id"]
test_df = test_df.drop("Id", axis=1)

# Preprocess the test data
X_test_preprocessed = preprocess_test_data(test_df, preprocessor).astype(np.float32)

# Make predictions using the best model
test_predictions = best_model.predict(X_test_preprocessed).astype(np.float32)

# Create the output DataFrame
output_df = pd.DataFrame({"Id": ids, "SalePrice": test_predictions})

# Save the predictions to a CSV file
output_file_path = "pytorch_grid_search_predictions.csv"
write_predictions(ids, test_predictions, output_file_path)

print(f"Predictions saved to {output_file_path}")
