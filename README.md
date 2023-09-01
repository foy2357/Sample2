# Data Processing, CNN Classification, and Data Statistics Scripts

This repository contains three Python scripts for various tasks related to data processing, Convolutional Neural Network (CNN) classification, and data statistics analysis. Each script serves a specific purpose and can be used in different stages of a machine learning project.

## Scripts

1. `data_processing_and_merge.py`
    - **Purpose**: This script is used for data preprocessing and merging. It reads multiple CSV files, preprocesses them, performs one-hot encoding, and pads the data to a specified length. Finally, it combines the processed data and saves it to CSV files.
    - **Usage**: Modify the script to specify the input and output directories and run it to process and merge your data.

2. `cnn_classification.py`
    - **Purpose**: This script is designed for training and evaluating a Convolutional Neural Network (CNN) classification model. It assumes that the data has already been preprocessed and padded. It defines the CNN model, compiles it, and trains it on the data. It also includes evaluation and visualization of the model's performance.
    - **Usage**: Customize the script to match your data and model requirements, then run it to train and evaluate the CNN model.

3. `csv_data_statistics.py`
    - **Purpose**: This script is used to analyze and visualize statistics related to CSV data. It calculates class frequencies, creates bar graphs, box plots with mean values, and histograms. It also computes statistics such as maximum, minimum, mean, median, variance, and standard deviation of epoch counts in the data.
    - **Usage**: Modify the script to specify the input CSV files and run it to perform data statistics analysis.

## Prerequisites

Before using these scripts, ensure you have the following dependencies installed:
- Python 3.10
- NumPy
- pandas
- Matplotlib
- Seaborn
- TensorFlow (for `cnn_classification.py`, if not already installed)
