import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from data import *

dataset_df = load_dataset(TRAIN_FILE_PATH)
X_preprocessed, y, preprocessor = preprocess_features(dataset_df)

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)


# Define the parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True],
    # "n_estimators": [500],
    # "max_depth": [10],
    # "min_samples_split": [2],
    # "min_samples_leaf": [1],
    # "bootstrap": [True],
}

# Initialize the Random Forest model
model = RandomForestRegressor(n_jobs=-1, random_state=42)

# Initialize the Grid Search
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)

# Fit the Grid Search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Use the best model
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

test_df = pd.read_csv(TEST_FILE_PATH)
ids = test_df["Id"]
test_df = test_df.drop("Id", axis=1)

X_test_preprocessed = preprocess_test_data(test_df, preprocessor)

test_predictions = model.predict(X_test_preprocessed)

output_file_path = "grid_search_random_forest_predictions.csv"
write_predictions(ids, test_predictions, output_file_path)
