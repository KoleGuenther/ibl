import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests


def load_and_prepare_data(file_path):
    """
    Update Log:
    10/15/24 Changed variable names to make it more clear and understandable. Also changed column names that it splits into as 'lat' and 'long' instead of 'latitude' and 'longitude'. Added Docstrings

    Function: Splits the 'latlong' column of the csv file into 'lat' and 'long' columns, splitting at the comma.

    Parameter:
        - file_path: The dataset that needs to be loaded and prepared.

    Return: Splits a 'latlong' column if it exists, otherwise skips this step. Prints a statement about what happened.
    """
    # Load dataset
    dataset = pd.read_csv(file_path)

    # Check if 'latlong' column exists before attempting to split it
    if 'latlong' in dataset.columns:
        # Split latlong into separate columns
        dataset[['lat', 'long']] = dataset['latlong'].str.split(',', expand=True).astype('float')
        dataset = dataset.drop(['latlong'], axis=1)
        print("The 'latlong' column was split into 'lat' and 'long'")
    else:
        print("No 'latlong' column found. Skipping split step.")

    return dataset


def add_elevation_data(pre_elevation_dataset):
    """
    Update Log:
    10/15/24 Added more accurate variable names and parameters so that it is more understandable. Added Docstrings

    Function: Uses dataset with lat and long to add elevation data to the dataset using the Open-Meteo API.

    Returns: pd.DataFrame: The dataset with elevation data added renamed as 'prepped_dataset'.
    """
    lat_list = pre_elevation_dataset['lat'].tolist()
    long_list = pre_elevation_dataset['long'].tolist()
    elevation = np.array([])

    for i in range(0, len(lat_list) - 100, 100):
        lat_listx = str(lat_list[i:i + 100]).strip('[]').replace(' ', '')
        long_listx = str(long_list[i:i + 100]).strip('[]').replace(' ', '')
        try:
            r = requests.get(f'https://api.open-meteo.com/v1/elevation?latitude={lat_listx}&longitude={long_listx}')
            result = r.json()
            if 'elevation' not in result:
                print(f"Unexpected response format for batch {i}: {result}")
                elevation = np.append(elevation, [np.nan] * 100)
            else:
                elevation = np.append(elevation, result['elevation'])
        except Exception as e:
            print(f"Exception for batch {i}: {e}")
            elevation = np.append(elevation, [np.nan] * 100)

    print(f"Processed batch {i}, elevation length: {len(elevation)}")

    remaining = len(lat_list) % 100
    if remaining > 0:
        lat_listx = str(lat_list[-remaining:]).strip('[]').replace(' ', '')
        long_listx = str(long_list[-remaining:]).strip('[]').replace(' ', '')
        try:
            r = requests.get(f'https://api.open-meteo.com/v1/elevation?latitude={lat_listx}&longitude={long_listx}')
            result = r.json()
            if 'elevation' not in result:
                print(f"Unexpected response format for last batch: {result}")
                elevation = np.append(elevation, [np.nan] * remaining)
            else:
                elevation = np.append(elevation, result['elevation'])
        except Exception as e:
            print(f"Exception for last batch: {e}")
            elevation = np.append(elevation, [np.nan] * remaining)

    # Ensure length consistency
    if len(elevation) < len(pre_elevation_dataset):
        elevation = np.append(elevation, [np.nan] * (len(pre_elevation_dataset) - len(elevation)))
    elif len(elevation) > len(pre_elevation_dataset):
        elevation = elevation[:len(pre_elevation_dataset)]

    pre_elevation_dataset['elevation'] = elevation
    pre_elevation_dataset['elevation'] = pre_elevation_dataset['elevation'].fillna(
        pre_elevation_dataset['elevation'].mean())

    # Update variable name to indicate that the dataset is prepped - Readability
    prepped_dataset = pre_elevation_dataset

    return prepped_dataset


def handle_missing_values(prepped_dataset, missing_value_placeholder):
    """
    Update Log:
    10/15/24 Moved from main.py to make program more readable (and shorter) for the rest of the team. Made variable names and parameters readable as well. Added Docstrings

    Function: Replaces the specified placeholder values in the dataset with NaN, which then gets replaced with the mean of each column. This prevents errors and gets rid of extreme values that might sway the model.

    Parameters
        - prepped_dataset: The prepped dataset.
        - missing_value_placeholder: The missing value that will be replaced (normally -99999)

    """
    # Drop unnamed columns that are completely empty
    prepped_dataset = prepped_dataset.loc[:, ~prepped_dataset.columns.str.contains('^Unnamed')]

    # Drop columns with all NaN values
    prepped_dataset = prepped_dataset.dropna(axis=1, how='all')

    # Count # of placeholders before replacement
    num_placeholders = (prepped_dataset == missing_value_placeholder).sum().sum()

    # Replace placeholder values with NaN
    prepped_dataset.replace(missing_value_placeholder, np.nan, inplace=True)

    # Count how many NaNs are present after replacement
    num_nans = prepped_dataset.isna().sum().sum()
    prepped_dataset.fillna(prepped_dataset.mean(), inplace=True)

    # Print replacement information
    print(f"Number of placeholder values ({missing_value_placeholder}) replaced with NaN: {num_placeholders}")
    print(f"Number of NaN values filled with column mean: {num_nans}")

    # Renames the variable, so it is clear that this is a cleaned version AFTER the dataset has looked through, fixing the missing value that was indicated in the parameters.
    cleaned_dataset = prepped_dataset

    return cleaned_dataset


def scale_features(cleaned_dataset, feature_columns):
    """
    Update Log:
    10/15/24 Moved from main.py to try and make program more readable for rest of the team. Changed variable names so that it is easier to understand. Added Docstrings

    Function: Scales the data from the specified columns (parameter feature_columns) so that there is a mean of 0 and a standard deviation of 1, making the model training more efficient and consistent.

    Parameters:
        - cleaned_dataset: The dataset that has been prepped and cleaned.
        - feature_columns: A list of variable names that need to be scaled.

    Returns: A new dataset containing the scaled variables with the original feature column names, and keeps the same indices so that it aligns with the rest of the dataset.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cleaned_dataset[feature_columns])
    scaled_dataset = pd.DataFrame(scaled_features, columns=feature_columns, index=cleaned_dataset.index)

    return scaled_dataset
