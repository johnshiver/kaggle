import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TRAIN_FILE_PATH = (
    "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/train.csv"
)

TEST_FILE_PATH = (
    "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/test.csv"
)


def load_dataset(train_file_path):
    """Load the dataset from the specified file path."""
    df = pd.read_csv(train_file_path)
    df = df.drop("Id", axis=1)
    sparse_cols = [
        "LotFrontage",
        "Alley",
        "FireplaceQu",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "PoolArea",
        "MasVnrType",
        "LotShape",
        "Street",
    ]
    df = df.drop(columns=sparse_cols)
    return df


def remove_outliers(df, column):
    """Remove outliers from the specified column in the DataFrame."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def preprocess_features(df):
    """Preprocess the features of the dataset."""
    df = remove_outliers(df, "GrLivArea")
    df = remove_outliers(df, "TotalBsmtSF")
    df = remove_outliers(df, "1stFlrSF")

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=(["object"])).columns

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor


def preprocess_test_data(test_df, preprocessor):
    """Preprocess the test data using the fitted preprocessor."""
    test_df["TotalSF"] = (
        test_df["TotalBsmtSF"] + test_df["1stFlrSF"] + test_df["2ndFlrSF"]
    )
    test_df["TotalBath"] = (
        test_df["FullBath"]
        + (0.5 * test_df["HalfBath"])
        + test_df["BsmtFullBath"]
        + (0.5 * test_df["BsmtHalfBath"])
    )
    X_test_preprocessed = preprocessor.transform(test_df)
    return X_test_preprocessed


def write_predictions(ids, predictions, output_file_path):
    """Write the predictions to a CSV file."""
    output_df = pd.DataFrame({"Id": ids, "SalePrice": predictions})
    output_df.to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")
