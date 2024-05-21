import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score

# Load the dataset
train_file_path = "/Volumes/HDD2/datasets/titanic/train.csv"
test_file_path = "/Volumes/HDD2/datasets/titanic/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Create new features
# train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
# test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
# train_df["IsAlone"] = (train_df["FamilySize"] == 1).astype(int)
# test_df["IsAlone"] = (test_df["FamilySize"] == 1).astype(int)

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

# Define the parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Initialize the Random Forest model
model = RandomForestClassifier(n_jobs=-1, random_state=42)

# Initialize the Grid Search
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)

# Fit the Grid Search to the data
grid_search.fit(X_train_preprocessed, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Use the best model
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train_preprocessed, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train_preprocessed)
y_pred_test = best_model.predict(X_test_preprocessed)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Cross-Validation
cv_scores = cross_val_score(best_model, X_train_preprocessed, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Preprocess the test data
X_test_preprocessed_final = preprocessor.transform(test_df)

# Make predictions using the best model
test_predictions = best_model.predict(X_test_preprocessed_final)

# Create the output DataFrame
output_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": test_predictions}
)

# Save the predictions to a CSV file
output_file_path = "titanic_predictions.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
