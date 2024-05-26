import pandas as pd


def load_data(file_path):
    """
    Load the data from a CSV file.

    :param file_path: Path to the CSV file
    :return: DataFrame containing the data
    """
    return pd.read_csv(file_path)


def inspect_data(df):
    """
    Perform basic introspection of the data.

    :param df: DataFrame containing the data
    :return: None
    """
    # Display the first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    # Display the data types of each column
    print("\nData types of each column:")
    print(df.dtypes)

    # Display basic statistics of the dataset
    print("\nBasic statistics of the dataset:")
    print(df.describe())

    # Display the number of missing values in each column
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Display the shape of the dataset
    print("\nShape of the dataset:")
    print(df.shape)


def column_info(df):
    """
    Display information about each column in the DataFrame.

    :param df: DataFrame containing the data
    :return: None
    """
    for column in df.columns:
        print(f"\nColumn: {column}")
        print(f"  Unique values: {df[column].nunique()}")
        print(f"  Sample values: {df[column].unique()[:5]}")
        print(f"  Data type: {df[column].dtype}")
        print(f"  Null values: {df[column].isnull().sum()}")


# Load the data
file_path = "~/datasets/LANL-Earthquake-Prediction/train.csv"
data = load_data(file_path)

# Perform basic introspection
inspect_data(data)

# Display detailed column information
column_info(data)
