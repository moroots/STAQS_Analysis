# -*- coding: utf-8 -*-
"""
Created on 2026-03-02 12:49:50

@author: Maurice Roots

Description:
     - A short script for Datacleaning
     - May turn into module later...
"""

#%%
 
import pickle
from atmoz.resources.path_manager import PathManager
from pydantic import BaseModel
from typing import List, Dict, Any

import pandas as pd


data_dir = PathManager("../paths.json").get_path("data_DESKTOP")


from atmoz.resources.importData import HSRL2, TOLNet, read_ict, h5Dataset_timestamp, to_df
from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path
import numpy as np
import matplotlib.dates as mdates

def main_import(dir_path: Path): 
    if not isinstance(dir_path, Path): 
        dir_path = Path(dir_path)

    tolnet_data = TOLNet()
    tolnet_data.import_data( list( (dir_path / r"TOLNet_hdf5").glob("*.h5") ) )

    tolnet = {}
    for key, dataset in tolnet_data.data.items():

        z = dataset['ALTITUDE'][()].astype(float)
        lat = dataset['LATITUDE.INSTRUMENT'][()].astype(float)
        lon = dataset['LONGITUDE.INSTRUMENT'][()].astype(float)
        elev = dataset['ALTITUDE.INSTRUMENT'][()].astype(float)
        integration_time = dataset['INTEGRATION.TIME'][()].astype(float)
        
        start_timestamps = h5Dataset_timestamp(dataset['DATETIME.START'])
        stop_timestamps = h5Dataset_timestamp(dataset['DATETIME.STOP'])

        geometry = [Point(xy) for xy in zip(lon, lat)] * len(start_timestamps)

        ozone_array = dataset['O3.MIXING.RATIO.VOLUME_DERIVED'][()].astype(float)
        uncertainty_array = dataset['O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.RANDOM.STANDARD'][()].astype(float)
        uncertainty_array[uncertainty_array <= -999] = np.nan
        ozone_array[ozone_array <= -999] = np.nan 

        units = dataset['O3.MIXING.RATIO.VOLUME_DERIVED'].attrs.get("units", "")
        
        ozone_df = to_df(ozone_array, start_timestamps, z)

        temp = {}
        temp["df"] = gpd.GeoDataFrame(ozone_df, geometry=geometry, crs="EPSG:4326")
        temp["start_time"] = start_timestamps
        temp["stop_time"] = stop_timestamps
        temp["uncertainty"] = uncertainty_array
        temp["elevation"] = elev
        temp["integration_time"] = integration_time
        temp["units"] = units
        temp["geometry"] = geometry
        temp["altitude"] = z

        instrument_name = key.split("_")[2] + "_" + key.split("_")[3]
        if instrument_name not in tolnet.keys():
            tolnet[instrument_name] = {}
        tolnet[instrument_name].update({key: temp})

    return tolnet

def fill_time_gaps(df, start_time, stop_time):

    df = df.copy()

    columns = df.columns
    timestamps = []
    values = []

    # estimate typical integration step
    durations = (stop_time - start_time).round("1min")
    unique_timedeltas, counts = np.unique(durations, return_counts=True)

    diff = pd.Timedelta(unique_timedeltas[np.argmax(counts)])

    # first record
    timestamps.append(start_time[0])
    values.append(df.iloc[0])

    for i in range(1, len(start_time)):

        gap = start_time[i] - stop_time[i-1]

        if gap.round("1min") > diff:

            gap_times = pd.date_range(
                stop_time[i-1] + diff,
                start_time[i] - diff,
                freq=diff
            )

            for t in gap_times:
                timestamps.append(t)
                values.append(pd.Series(np.nan, index=columns))

        timestamps.append(start_time[i])
        values.append(df.iloc[i])

    return pd.DataFrame(values, index=pd.DatetimeIndex(timestamps), columns=columns)


tolnet = main_import(data_dir)


