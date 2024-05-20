import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
train_file_path = "/Volumes/HDD2/datasets/titanic/train.csv"
test_file_path = "/Volumes/HDD2/datasets/titanic/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Separate target from features
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Preprocess the testing data
X_test_preprocessed = preprocessor.transform(X_test)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=512, n_jobs=-1, random_state=42)

# Train the model
model.fit(X_train_preprocessed, y_train)

# Make predictions
y_pred_train = model.predict(X_train_preprocessed)
y_pred_test = model.predict(X_test_preprocessed)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Preprocess the test data
X_test_preprocessed_final = preprocessor.transform(test_df)

# Make predictions using the Random Forest model
test_predictions = model.predict(X_test_preprocessed_final)

# Create the output DataFrame
output_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": test_predictions}
)

# Save the predictions to a CSV file
output_file_path = "titanic_predictions.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
