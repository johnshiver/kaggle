import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
train_file_path = (
    # "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/train.csv"
    "/media/johnshiver/hdd-fast/house-prices-advanced-regression-techniques/train.csv"
)
dataset_df = pd.read_csv(train_file_path)

# Drop the 'Id' column
dataset_df = dataset_df.drop("Id", axis=1)

# drop sparse columns
dataset_df = dataset_df.drop("LotFrontage", axis=1)
dataset_df = dataset_df.drop("Alley", axis=1)
dataset_df = dataset_df.drop("FireplaceQu", axis=1)
dataset_df = dataset_df.drop("PoolQC", axis=1)
dataset_df = dataset_df.drop("Fence", axis=1)
dataset_df = dataset_df.drop("MiscFeature", axis=1)

# Separate target from features
X = dataset_df.drop("SalePrice", axis=1)
y = dataset_df["SalePrice"]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing for numerical data: fill missing values with median and standardize
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Preprocessing for categorical data: fill missing values with most frequent and one-hot encode
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# Define the parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
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

# Create the output file with predictions
test_file_path = (
    # "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/test.csv"
    "/media/johnshiver/hdd-fast/house-prices-advanced-regression-techniques/test.csv"
)
test_df = pd.read_csv(test_file_path)
ids = test_df["Id"]
test_df = test_df.drop("Id", axis=1)

# Preprocess the test data
X_test_preprocessed = preprocessor.transform(test_df)

# Make predictions using the best model
test_predictions = best_model.predict(X_test_preprocessed)

# Create the output DataFrame
output_df = pd.DataFrame({"Id": ids, "SalePrice": test_predictions})

# Save the predictions to a CSV file
output_file_path = "predictions.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
