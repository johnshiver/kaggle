import pandas as pd
import matplotlib.pyplot as plt


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


def visualize_data(df):
    """
    Visualize the acoustic_data and time_to_failure columns.

    :param df: DataFrame containing the data
    :return: None
    """
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df["acoustic_data"][:50000])
    plt.title("Acoustic Data (first 50,000 samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Acoustic Data")

    plt.subplot(2, 1, 2)
    plt.plot(df["time_to_failure"][:50000])
    plt.title("Time to Failure (first 50,000 samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Time to Failure")

    plt.tight_layout()
    plt.show()


def summary_statistics(df):
    """
    Compute and print summary statistics for acoustic_data and time_to_failure columns.

    :param df: DataFrame containing the data
    :return: None
    """
    stats = {
        "acoustic_data": {
            "mean": df["acoustic_data"].mean(),
            "median": df["acoustic_data"].median(),
            "variance": df["acoustic_data"].var(),
        },
        "time_to_failure": {
            "mean": df["time_to_failure"].mean(),
            "median": df["time_to_failure"].median(),
            "variance": df["time_to_failure"].var(),
        },
    }

    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"{key}:")
        for stat, val in value.items():
            print(f"  {stat}: {val}")


def correlation_analysis(df):
    """
    Perform correlation analysis between acoustic_data and time_to_failure.

    :param df: DataFrame containing the data
    :return: None
    """
    correlation = df["acoustic_data"].corr(df["time_to_failure"])
    print(f"\nCorrelation between acoustic_data and time_to_failure: {correlation}")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["acoustic_data"][:50000], df["time_to_failure"][:50000], alpha=0.1)
    plt.title("Correlation between Acoustic Data and Time to Failure")
    plt.xlabel("Acoustic Data")
    plt.ylabel("Time to Failure")
    plt.show()


# Load the data
file_path = "~/datasets/LANL-Earthquake-Prediction/train.csv"
data = load_data(file_path)

# Perform basic introspection
inspect_data(data)

# Display detailed column information
column_info(data)

# Visualize the data
visualize_data(data)

# Compute summary statistics
summary_statistics(data)

# Perform correlation analysis
correlation_analysis(data)