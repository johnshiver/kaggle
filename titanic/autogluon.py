import pandas as pd
from autogluon.tabular import TabularPredictor

# Load the dataset
train_file_path = "/Volumes/HDD2/datasets/titanic/train.csv"
test_file_path = "/Volumes/HDD2/datasets/titanic/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Define the label column
label = "Survived"

# Train the model
predictor = TabularPredictor(label=label, eval_metric="accuracy").fit(
    train_data=train_df
)

# Evaluate the model on the training set
train_performance = predictor.evaluate(train_df)
print(f"Train Performance: {train_performance}")

# Make predictions on the test set
test_predictions = predictor.predict(test_df)

# Create the output DataFrame
output_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": test_predictions}
)

# Save the predictions to a CSV file
output_file_path = "autogluon_titanic_predictions.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
