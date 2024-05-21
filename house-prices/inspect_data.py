import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


def load_dataset(filepath):
    return pd.read_csv(filepath)


def summary_statistics(df):
    print("Summary Statistics:")
    print(df.describe())


def missing_values_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()


def correlation_matrix(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr_matrix = numeric_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def pairplot(df, features):
    sns.pairplot(df[features])
    plt.show()


def distribution_plot(df, feature):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} Distribution")
    plt.show()


def box_plot(df, x_feature, y_feature):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=x_feature, y=y_feature, data=df)
    plt.title(f"{y_feature} vs {x_feature}")
    plt.show()


def count_plot(df, feature):
    plt.figure(figsize=(12, 8))
    sns.countplot(x=feature, data=df)
    plt.title(f"{feature} Count Plot")
    plt.xticks(rotation=90)
    plt.show()


def feature_importance_plot(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()


def main():
    train_file_path = (
        # "/media/johnshiver/hdd-fast/house-prices-advanced-regression-techniques/train.csv"
        "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/train.csv"
    )
    df = load_dataset(train_file_path)

    # Drop the 'Id' column
    df = df.drop("Id", axis=1)

    summary_statistics(df)
    missing_values_heatmap(df)
    correlation_matrix(df)

    # List of features to include in the pairplot
    pairplot_features = [
        "SalePrice",
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
    ]
    pairplot(df, pairplot_features)

    distribution_plot(df, "SalePrice")
    distribution_plot(df, "GrLivArea")

    box_plot(df, "OverallQual", "SalePrice")
    count_plot(df, "Neighborhood")
    box_plot(df, "Neighborhood", "SalePrice")

    # Prepare data for feature importance plot
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Handle categorical data (simple encoding for visualization purposes)
    X = pd.get_dummies(X, drop_first=True)

    feature_importance_plot(X, y)


if __name__ == "__main__":
    main()
