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

import matplotlib.pyplot as plt

import pandas as pd

data_dir = PathManager("../paths.json").get_path("data_DESKTOP")
fig_dir = PathManager("../paths.json").get_path("figures_GDRIVE")

from atmoz.resources.importData import HSRL2, TOLNet, read_ict, h5Dataset_timestamp, to_df
from atmoz.resources import makePlots 
from atmoz.resources import useful_functions, colorbars, plot_utilities, default_plot_params

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

def fill_time_gaps(df, start_time, stop_time, integration_time):

    df = df.copy()

    columns = df.columns
    timestamps = []
    values = []

    # estimate typical integration step
    diff = pd.Timedelta(integration_time, unit="h")

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


def plot_curtain_better(lidar, sonde=None, **kwargs):
    tz = kwargs.pop("tz", "UTC")
    params = useful_functions.merge_dicts(default_plot_params.tolnet_plot_params, kwargs)
    
    cmap, norm = colorbars.tolnet_ozone()

    if not isinstance(lidar["X"], list):
        lidar = {key: [lidar[key]] for key in lidar.keys()}
    
    with plt.rc_context(default_plot_params.curtain_plot_theme):
        fig, ax = plt.subplots()

        for X, Y, C in zip(lidar["X"], lidar["Y"], lidar["C"]):
            if isinstance(X, list) and isinstance(Y, list) and isinstance(C, list):
                for i in range(len(X)):
                    im = ax.pcolormesh(X[i], Y[i], C[i], cmap=cmap, norm=norm, shading="nearest", alpha=1)
            else: 
                im = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm, shading="nearest", alpha=1)
        
        if sonde: 
            ax.scatter(
                x = sonde["X"],
                y = sonde["Y"],
                c = sonde["C"],
                s = 100,
                cmap = cmap, 
                norm = norm,
                **sonde.get("scatter_params", {})
                )

            vline_params = sonde.get("vline_params", {})
            skip = vline_params.pop("skip", 10) if vline_params and "skip" in vline_params else 10

            ax.vlines(sonde["X"][::skip]-pd.Timedelta(10, unit='min'), ymin=sonde["Y"][::skip] - 0.05, ymax=sonde["Y"][::skip] + 0.05,
                    **vline_params)

            ax.vlines(sonde["X"][::skip]+pd.Timedelta(10, unit='min'), ymin=sonde["Y"][::skip] - 0.05, ymax=sonde["Y"][::skip] + 0.05,
                    **vline_params)

        params["fig.colorbar"]["mappable"] = im

        plot_utilities.apply_datetime_axis(ax, tz=tz)
        
        plot_utilities.apply_plot_params(fig, ax, **params)

        plt.close()
    return 


#%%

# for instrument_name in tolnet.keys():
#     dates = 
#%%
import matplotlib.dates as mdates
for instrument_name in tolnet.keys():
    date_start = []
    date_stop = []
    X = []; Y = []; C = []
    for filename in tolnet[instrument_name].keys():
        df = fill_time_gaps(
                tolnet[instrument_name][filename]["df"].copy().drop(columns=["geometry"]),
                start_time = tolnet[instrument_name][filename]["start_time"],
                stop_time = tolnet[instrument_name][filename]["stop_time"],
                integration_time = tolnet[instrument_name][filename]["integration_time"][0]
                ).sort_index(axis=0).sort_index(axis=1) 

        X.append(df.index)
        Y.append(df.columns.astype(float) / 1000)
        C.append(df.values.T * 1000)

        date_start.append(df.index.min())
        date_stop.append(df.index.max())

    lidar = {
        "X": X,
        "Y": Y,
        "C": C
    }

    date_start = np.datetime64("2023-07-25")
    date_stop = np.datetime64("2023-08-16")

    title = f"{instrument_name} [{date_start} - {date_stop}]"

    folder = fig_dir / "instrument_full"
    folder.mkdir(parents=True, exist_ok=True)

    savename = title.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '').replace('-', '_') + ".png"

    params = {
        "fig.colorbar": {
            "label": "Ozone Mixing Ratio (ppbv)"
            },
        "ax.set_xlim": [date_start, date_stop],
        "ax.set_autoscale_on": False,
        "ax.set_ylim": [0, 10],
        "ax.set_title": title,
        "fig.savefig": {
            "fname": folder / savename,
            "format": "png",
            "dpi": 600,
            "transparent": True
            },
        "tz": "America/New_York"
        }

    plot_curtain_better(lidar, **params)

# %%
