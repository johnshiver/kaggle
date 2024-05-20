import pandas as pd
from autogluon.tabular import TabularPredictor

# Load the dataset
train_file_path = (
    # "/media/johnshiver/hdd-fast/house-prices-advanced-regression-techniques/train.csv"
    "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/train.csv"
)
dataset_df = pd.read_csv(train_file_path)

# Drop the 'Id' column
dataset_df = dataset_df.drop("Id", axis=1)

# Separate target from features
X = dataset_df.drop("SalePrice", axis=1)
y = dataset_df["SalePrice"]

# Combine features and target into one dataframe for AutoGluon
train_data = pd.concat([X, y], axis=1)

# Initialize and train the AutoGluon model
predictor = TabularPredictor(
    label="SalePrice", eval_metric="root_mean_squared_error"
).fit(train_data)

# Evaluate the model
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard)

# Load the test dataset
test_file_path = (
    # "/media/johnshiver/hdd-fast/house-prices-advanced-regression-techniques/test.csv"
    "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/train.csv"
)
test_df = pd.read_csv(test_file_path)
ids = test_df["Id"]

# Make predictions on the test dataset
predictions = predictor.predict(test_df)

# Create the output DataFrame
output_df = pd.DataFrame({"Id": ids, "SalePrice": predictions})

# Save the predictions to a CSV file
output_file_path = "predictions.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
