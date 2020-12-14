"""
This module contains functions used for the pre-processing of data before
use for model training, validation, testing, or predictions.
"""
from typing import Tuple
import datetime

import xarray as xr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from data_classes import DataFrameSet


FEATURE_COORD_VARS = ('longitude', 'latitude', 'time')
DEFAULT_VARS = ('u10', 'd2m', 'fal', 'tcc', 'swvl1', 'stl1', 'v10', 't2m', 'sp', 'tcwv',
                'h2m')


def ingest_new_data(cache_data: bool = False, normalize: bool = True,
                    features_data_dir: str = "data/daily_values/*.nc",
                    cache_dir: str = "data/cache/",
                    split_shape: Tuple[float, float, float] = (.7, .15, .15)
                    ) -> DataFrameSet:
    """
    Returns four dataframes. One always just a cleaned dataframe, a training, validation
    and testing dataframe and a cleaned dataframe that may be normalized. This is the default
    ingest function. Loads data from fresh netCDF files with an option to cache it after basic
    cleaning. Further dataprocessing options avaliable
    to split the dataset in different ways and to normalize it.

    - len(split_shape) == 3
    - sum(list(split_shape)) == 1
    """

    ds = xr.open_mfdataset(features_data_dir)

    calculate_h2m(ds)

    ds = average_dataset(ds)

    if cache_data:
        save_netcdf(ds, path=cache_dir)

    ds = to_data_frame(ds)

    train_df, val_df, test_df, ground_df = split_normalize_dataframe(
        ds, normalize=normalize, split_shape=split_shape)

    dfs = DataFrameSet(train_df, val_df, test_df, ground_df)

    print('New data ingest complete.')

    return dfs


def ingest_cached_data(filename: str, normalize: bool = True,
                       cleaned_dir: str = "data/cache/",
                       split_shape: Tuple[float, float, float] = (.7, .15, .15)
                       ) -> DataFrameSet:
    """
    Returns four dataframes. One always just a cleaned dataframe, a training, validation
    and testing dataframe and a cleaned dataframe that may be normalized. This is the default
    ingest function. Loads from an already cleaned netCDF file. Further dataprocessing options
    to split the dataset in different ways and to normalize it.

    - len(split_shape) == 3
    - sum(list(split_shape)) == 1
    """
    ds = xr.open_mfdataset(cleaned_dir + filename + '.nc')

    ds = to_data_frame(ds)

    train_df, val_df, test_df, ground_df = split_normalize_dataframe(
        ds, normalize=normalize, split_shape=split_shape)

    dfs = DataFrameSet(train_df, val_df, test_df, ground_df)

    print('Cached data ingest complete.')
    return dfs


def calculate_h2m(dataset: xr.core.dataset.Dataset) -> None:
    """
    Mutates dataset with 2 metre relative humidity based on present
    information on 2 metre dewpoint temperature and 2 metre temperature.

    - 't2m' in dataset.variables
    - 'd2m' in dataset.variables
    """
    attrs = {'units': '%', 'long_name': '2 metre relative humidity'}
    dataset['h2m'] = 100 * dataset['d2m'] / dataset['t2m']
    dataset['h2m'].attrs = attrs


def clean_dataset(dataset: xr.core.dataset.Dataset,
                  feature_coord_vars: tuple = FEATURE_COORD_VARS,
                  feature_vars: tuple = DEFAULT_VARS) -> xr.core.dataset.Dataset:
    """
    Mutates dataset such that only necessary information remains.
    """
    for var in dataset.variables:
        if (var not in feature_coord_vars) and (var not in feature_vars):
            dataset = dataset.drop(var)
    return dataset


def crop_dataset(dataset: xr.core.dataset.Dataset,
                 min_lon: float, max_lon: float,
                 min_lat: float, max_lat: float) -> xr.core.dataset.Dataset:
    """
    Returns a cropped dataset of a specific location

    - min_lon < max_lon
    - min_lat < max_lat
    - -180 < min_lon < 180
    - -180 < max_lon < 180
    - -90 < min_lat < 90
    - -90 < max_lat < 90
    """
    mask_lon = (dataset.longitude >= min_lon) & (dataset.longitude <= max_lon)
    mask_lat = (dataset.latitude >= min_lat) & (dataset.latitude <= max_lat)

    cropped_ds = dataset.where(mask_lon & mask_lat, drop=True)
    return cropped_ds


def average_dataset(dataset: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    """
    Returns dataset with all variables averaged across avaliable longitude and latitude.
    """
    for var in dataset.data_vars:
        dataset[var] = dataset[var].mean(('longitude', 'latitude'))

    dataset = dataset.drop((('longitude', 'latitude')))
    return dataset


def save_netcdf(dataset: xr.core.dataset.Dataset, path: str = "data/cache/") -> None:
    """
    Write dataset contents as a netCDF file.
    """
    dataset.to_netcdf(
        path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.nc')


def check_frequency_features(dataframe: xr.core.dataset.Dataset,
                             feature: str) -> None:
    """
    Plots a feature's real-valued fourier transformation over time.

    - feature in dataframe.columns
    """
    fft = tf.signal.rfft(dataframe[feature])

    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(dataframe[feature])
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples_h / hours_per_year

    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 400000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')


def split_normalize_dataframe(df: pd.core.frame.DataFrame,
                              split_shape: tuple = (.7, .15, .15),
                              normalize: bool = True) -> pd.core.frame.DataFrame:
    """
    Returns normalized dataframe split into groups for training, validation, and testing.

    - len(split_shape) == 3
    - sum(list(split_shape)) == 1
    """
    print(f"SPLITTING dataset into {split_shape}")
    n = len(df)

    index = [int(n * split_shape[0]),
             int(n * (split_shape[0] + split_shape[1]))]

    train_df = df[0:index[0]]
    val_df = df[index[0]:index[1]]
    test_df = df[index[1]:]

    if normalize:
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        df = (df - train_mean) / train_std
    return train_df, val_df, test_df, df


def to_data_frame(ds: xr.core.dataset.Dataset,
                  na_handeling: str = 'fill') -> pd.core.frame.DataFrame:
    """
    Reterns a pandas DataFrame converted from an xarray dataset. Optionally handles nan values.

    - na_handeling is in ['fill', 'drop'] or na_handeling == None
    """
    ds = {key: np.array(value) for key, value in dict(ds).items()}

    ds = pd.DataFrame.from_dict(ds)
    if na_handeling == 'fill':
        ds = ds.fillna(0)
    elif na_handeling == 'drop':
        ds = ds.dropna(axis='columns')

    return ds


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        # the names (strs) of imported modules
        'extra-imports': ['xarray', 'pandas', 'datetime',
                          'tensorflow', 'matplotlib.pyplot',
                          'numpy', 'python_ta.contracts',
                          'data_classes'],
        'allowed-io': ['split_normalize_dataframe', 'load_and_merge_co2',
                       'ingest_new_data', 'ingest_cached_data'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
