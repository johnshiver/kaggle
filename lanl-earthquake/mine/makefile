# Makefile for downloading the LANL-Earthquake-Prediction dataset from Kaggle

# Define variables
COMPETITION_NAME = LANL-Earthquake-Prediction
DATASET_PATH = lanl-earthquake

# Default target
.PHONY: all
all: download-dataset

# Target to download the dataset from Kaggle
.PHONY: download-dataset
download-dataset:
	kaggle competitions download -c $(COMPETITION_NAME) -p $(DATASET_PATH)