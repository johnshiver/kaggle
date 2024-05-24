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


def load_data(file_path):
    """Load the training and testing datasets."""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(train_df):
    """Preprocess the data: separate target, handle missing values, encode categorical variables, and standardize numerical variables."""
    # Separate target from features
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    cols_to_drop = [
        "Name",
    ]
    X = X.drop(columns=cols_to_drop)

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Preprocessing for numerical data: fill missing values with median and standardize
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
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

    return (
        X_train_preprocessed,
        X_test_preprocessed,
        y_train,
        y_test,
    )
