import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


"""
Notes

Name shouldnt affect anything, can drop

Columns with missing data:
 - Cabin
 - Age

 Survied correlates most with Fare

"""


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


def show_distributions(df):
    """Inspect the distribution of each feature in the dataset."""
    for column in df.columns:
        plt.figure(figsize=(10, 6))

        if df[column].dtype in ["int64", "float64"]:
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f"Distribution of {column}")
        else:
            sns.countplot(y=df[column].dropna())
            plt.title(f"Distribution of {column}")

        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()


def main():
    train_file_path = "/Volumes/HDD2/datasets/titanic/train.csv"
    df = load_dataset(train_file_path)

    summary_statistics(df)
    show_distributions(df)
    missing_values_heatmap(df)
    correlation_matrix(df)

    # feature_importance_plot(X, y)


if __name__ == "__main__":
    main()
