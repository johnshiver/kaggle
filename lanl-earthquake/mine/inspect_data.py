import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def downsample_data(df, factor):
    df_acoustic = (
        df["acoustic_data"]
        .groupby(np.arange(len(df)) // factor)
        .mean()
        .reset_index(drop=True)
    )
    df_time = (
        df["time_to_failure"]
        .groupby(np.arange(len(df)) // factor)
        .max()
        .reset_index(drop=True)
    )
    return pd.DataFrame({"acoustic_data": df_acoustic, "time_to_failure": df_time})


def visualize_data(df, threshold=0.01):
    """
    Visualize the acoustic_data and time_to_failure columns.
    Denote where an earthquake likely occurred with vertical lines.

    :param df: DataFrame containing the data
    :param threshold: Threshold for time_to_failure to denote earthquake occurrence
    :return: None
    """
    factor = 20
    df = downsample_data(df, factor)

    plt.figure(figsize=(15, 6))

    # Acoustic Data
    plt.subplot(2, 1, 1)
    plt.plot(df["acoustic_data"])
    # plt.plot(df["acoustic_data"])
    plt.title(f"Acoustic Data (downsample {factor})")
    plt.xlabel("Sample index")
    plt.ylabel("Acoustic Data")

    # Time to Failure
    plt.subplot(2, 1, 2)
    plt.plot(df["time_to_failure"])
    # plt.plot(df["time_to_failure"])
    plt.title(f"Time to Failure (downsample {factor})")
    plt.xlabel("Sample index")
    plt.ylabel("Time to Failure")

    # Find where time_to_failure is below the threshold and draw vertical lines
    for i, ttf in enumerate(df["time_to_failure"]):
        if ttf < threshold:
            plt.axvline(x=i, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig(f"dataset_visual_downsize_{factor}")
    plt.show()


# def visualize_data(df):
#     """
#     Visualize the acoustic_data and time_to_failure columns.

#     :param df: DataFrame containing the data
#     :return: None
#     """
#     plt.figure(figsize=(15, 6))

#     plt.subplot(2, 1, 1)
#     plt.plot(df["acoustic_data"][:50000])
#     plt.title("Acoustic Data (first 50,000 samples)")
#     plt.xlabel("Sample index")
#     plt.ylabel("Acoustic Data")

#     plt.subplot(2, 1, 2)
#     plt.plot(df["time_to_failure"][:50000])
#     plt.title("Time to Failure (first 50,000 samples)")
#     plt.xlabel("Sample index")
#     plt.ylabel("Time to Failure")

#     plt.tight_layout()
#     plt.show()


def find_earthquake_events(df, threshold=0.01):
    """
    Print out indices where time_to_failure is near 0.

    :param df: DataFrame containing the data
    :param threshold: Threshold for time_to_failure to denote earthquake occurrence
    :return: None
    """
    earthquake_indices = []
    for i, ttf in enumerate(df["time_to_failure"]):
        if ttf < threshold:
            earthquake_indices.append(i)

    print(
        f"Earthquake likely occurred at the following indices (time_to_failure < {threshold}):"
    )
    for idx in earthquake_indices:
        print(f"Index: {idx}")


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


def aggregate_data(data, num_rows=150000, precision=3):
    """
    Aggregate the data by rounded time_to_failure and compute the average acoustic_data and count.

    :param file_path: Path to the CSV file containing the data.
    :param num_rows: Number of rows to analyze (default: 50000).
    :param precision: Decimal places to round time_to_failure for grouping (default: 3).
    :return: DataFrame with aggregated data.
    """
    data_subset = data.iloc[:num_rows]

    # Round the time_to_failure values to the specified precision
    data_subset["rounded_time_to_failure"] = data_subset["time_to_failure"].round(
        precision
    )

    aggregated_data = (
        data_subset.groupby("rounded_time_to_failure")
        .agg(
            avg_acoustic_data=("acoustic_data", "mean"), count=("acoustic_data", "size")
        )
        .reset_index()
    )

    return aggregated_data


# Load the data
# file_path = "~/datasets/LANL-Earthquake-Prediction/train.csv"
file_path = "LANL-Earthquake-Prediction/train.csv"

# file_path = "/media/johnshiver/hdd-fast/lanl-earthquake/train.csv"
# file_path = "lanl-earthquake/train.csv"
data = load_data(file_path)

# Perform basic introspection
# inspect_data(data)

# Display detailed column information
# column_info(data)

# Visualize the data
visualize_data(data)
# find_earthquake_events(data)

# Compute summary statistics
# summary_statistics(data)

# Perform correlation analysis
# correlation_analysis(data)

# aggregated_data = aggregate_data(data)
# print(aggregated_data.head(10))
