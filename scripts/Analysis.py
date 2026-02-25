# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:26:50 2025

@author: Magnolia
"""
#%% 

from pathlib import Path
import pandas as pd

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

from src.importData import TOLNet, HSRL2, read_ict
from src.timeConversions import h5Dataset_timestamp
from src.geoSlicing import nearest_obs

import pickle

#%%
# ============================== #
# Imorting Data
# ============================== #

data_dir = Path("./data")



# - HSRL2 - #
data_filepath = r"HSRL2.pickle"
if not (data_dir / data_filepath).is_file():
    hsrl2_data = HSRL2()
    hsrl2_data.import_data( list( (data_dir / r"HSRL2").glob("*R1.h5") ) )

    hsrl2 = {}
    for key, dataset in hsrl2_data.data.items():
        z = dataset.z[()]
        lat = dataset.lat[()]
        lon = dataset.lon[()]

        geometry = [Point(xy) for xy in zip(lon.flatten(), lat.flatten())]
        timestamps = h5Dataset_timestamp(dataset.time)

        ozone = dataset.DataProducts.O3[()]
        units = dataset.DataProducts.O3.attrs.get("units", "")
        hsrl2[key] = gpd.GeoDataFrame(ozone, columns=z, geometry=geometry, index=timestamps, crs="EPSG:4326")
    
    with open(data_dir / data_filepath, "wb") as f:
        pickle.dump(hsrl2, f)
else: 
    with open(data_dir / data_filepath, "rb") as f:
        hsrl2 = pickle.load(f)




# - TOLNet - #
data_filepath = r"TOLNet.pickle"
if not (data_dir / data_filepath).is_file():
    tolnet_data = TOLNet()
    tolnet_data.import_data( list( (data_dir / r"TOLNet_hdf5").glob("*.h5") ) )

    tolnet = {}
    for key, dataset in tolnet_data.data.items():

        z = dataset['ALTITUDE'][()].astype(float)
        lat = dataset['LATITUDE.INSTRUMENT'][()].astype(float)
        lon = dataset['LONGITUDE.INSTRUMENT'][()].astype(float)

        timestamps = h5Dataset_timestamp(dataset['DATETIME.START'])
        geometry = [Point(xy) for xy in zip(lon, lat)] * len(timestamps)

        ozone_array = dataset['O3.MIXING.RATIO.VOLUME_DERIVED'][()].astype(float)
        ozone_array[ozone_array < -999] = np.nan 

        if ozone_array.ndim == 1:
            if len(timestamps) == len(ozone_array):
                ozone_array = ozone_array[:, np.newaxis]  # make it (N,1)
            elif len(z) == len(ozone_array):
                ozone_array = ozone_array[np.newaxis, :]  # make it (1,N)
            else:
                raise ValueError(f"Cannot align 1D array in dataset {key}")

        if ozone_array.shape[0] == len(timestamps) and ozone_array.shape[1] == len(z):
            ozone_df = pd.DataFrame(ozone_array, columns=z, index=timestamps)
        elif ozone_array.shape[1] == len(timestamps) and ozone_array.shape[0] == len(z):
            ozone_df = pd.DataFrame(ozone_array.T, columns=z, index=timestamps)
        else:
            raise ValueError(f"Shape mismatch in dataset {key}: ozone_array {ozone_array.shape}, timestamps {len(timestamps)}, altitudes {len(z)}")

        ozone[ozone <= -999] = np.nan

        units = dataset['O3.MIXING.RATIO.VOLUME_DERIVED'].attrs.get("units", "")
        tolnet[key] = gpd.GeoDataFrame(ozone_df, geometry=geometry, crs="EPSG:4326")

    with open(data_dir / data_filepath, "wb") as f:
        pickle.dump(tolnet, f)
else: 
    with open(data_dir / data_filepath, "rb") as f:
        tolnet = pickle.load(f)



# - Sondes - #
data_filepath = r"sondes.pickle"
if not (data_dir / data_filepath).is_file(): 
    sondes_dict = read_ict(Path(r"./data/Sondes").glob("*.ict"))

    temp = []
    for key in sondes_dict.keys():
        temp.append(sondes_dict[key]["data"])

    concat = pd.concat(temp)

    concat = concat.mask(concat < -999, np.nan)

    concat.dropna(subset=["Latitude_deg", "Longitude_deg"], inplace=True)

    concat["geometry"] = [Point(xy) for xy in zip(concat["Longitude_deg"], concat["Latitude_deg"])]
    sondes = gpd.GeoDataFrame(concat, geometry="geometry", crs="EPSG:4326")

    with open(data_dir / data_filepath, "wb") as f: 
        pickle.dump(sondes, f)

else: 
    with open(data_dir / data_filepath, "rb") as f: 
        sondes = pickle.load(f)



# - Surface Data (AirNow) - #
data_filepath = r"airnow_ozone.pickle"
if not (data_dir / data_filepath).is_file(): 
    from atmoz.surface.AirNow import AirNow

    airnow = AirNow()
    airnow_data = airnow.import_data(
        date_start='2023-07-23', 
        date_end="2023-08-18", 
        BBOX=["-80.655479", "35.574398", "-72.086143", "41.415693"], 
        parameters=["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"]
        )

    surface_ozone = []
    for key in airnow_data[0]["OZONE"].keys():
        temp = airnow_data[1][airnow_data[1]["id"] == key]
        geometry = [Point(temp["Longitude"], temp["Latitude"])] * len(airnow_data[0]["OZONE"][key])
        new = airnow_data[0]["OZONE"][key].copy()
        new["geometry"] = geometry
        surface_ozone.append(new)

    surface_ozone = gpd.GeoDataFrame(pd.concat(surface_ozone), geometry="geometry")

    with open(data_dir / data_filepath, "wb") as f: 
        pickle.dump(surface_ozone, f)

else:
    with open(data_dir / data_filepath, "rb") as f: 
        surface_ozone = pickle.load(f)




#%% 

# ============================================================ #
# Making Figures                                            
# ============================================================ #

import src
from src.makePlots import site_map

import importlib
importlib.reload(src.makePlots)

test_surface = surface_ozone.copy()
test_tolnet = pd.concat(tolnet, axis=0)
test_tolnet.index.names = ["filename", "timestamp"]

test_hsrl = pd.concat(hsrl2, axis=0)
test_tolnet.index.names = ["filename", "timestamp"]

test_sondes = sondes.copy()

#%% 

radius_km = 10
joined_left, joined_right = nearest_obs(test_sondes, test_tolnet, radius_km) 


#%%

site_map({"sondes": test_sondes, "sondes_vs_tolnet": joined_left, "tolnet": test_tolnet, })

#%% Plotting

# Seperate by tolnet location, 
# do regression of all points (need to time bin first), \
# then do profile plot (with std fill between)

#%% 

import matplotlib.pyplot as plt

locations = joined_right.geometry.to_list()
for location in locations:
    temp = test_tolnet[
        test_tolnet.geometry == location
        ].copy()
    
    radius_km = 10
    joined_left, _ = nearest_obs(test_sondes, temp, radius_km) 

    sondes_temp = test_sondes[
        test_sondes.geometry.isin(joined_left.geometry)
        ].copy()
    
    # site_map({"sondes": sondes_temp, "tolnet": temp})

    temp.drop(columns = "geometry", inplace=True)
    temp.index = temp.index.droplevel(0)
    temp.sort_index(axis=0, inplace=True); temp.sort_index(axis=1, inplace=True)
    temp.dropna(axis=1, how="all", inplace=True)
    
    sondes_temp.drop(columns="geometry", inplace=True)
    break


from atmoz.resources import plot_utilities, useful_functions, colorbars, default_plot_params

diffs = temp.index.to_series().diff().dropna()
diffs[(diffs > pd.Timedelta((60*60*24), unit="s") ) | (diffs <= pd.Timedelta(0, unit="s") )] = np.nan
median_diff = round(diffs.min().seconds / 60, -1)


temp = temp.resample(f"{median_diff} min").mean().resample(f"{median_diff} min").ffill()

X = temp.index
Y = temp.columns.astype(float)
C = temp.values.T*1000

def plot_curtain(X, Y, C, **kwargs):
    params = useful_functions.merge_dicts(default_plot_params.tolnet_plot_params, kwargs)
    xlims = params.pop("xlims", "auto")
    time_resolution = params.pop("time_resolution", "auto")
    
    cmap, norm = colorbars.tolnet_ozone()

    with plt.rc_context(default_plot_params.curtain_plot_theme):
        fig, ax = plt.subplots()

        im = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm, shading="nearest", alpha=1)
        ax.scatter(x = sondes_temp.index, y = sondes_temp['Altitude_km'] *1000, c=sondes_temp["Ozone_ppbv"], cmap=cmap, norm=norm)
        
        params["fig.colorbar"]["mappable"] = im

        plot_utilities.apply_datetime_axis(ax)
        
        # plot_utilities.apply_plot_params(fig, ax, **params)

        plt.show()
    return 

plot_curtain(X, Y, C)

#%%
from scipy import stats

# Example variables
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.1, 2.1, 2.9, 4.05, 5.0])

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"R-squared: {r_value**2:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Standard error: {std_err:.3f}")

# Plot
plt.scatter(x, y, label="Data")
plt.plot(x, slope*x + intercept, color='red', label="Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# joined_left, joined_right = 
# temp = source.drop(columns="geometry", inplace=False)
# temp_avg = temp.mean(axis=0)
# temp_std = temp.std(axis=0, skipna=True)

# nan_mask = temp_std.notna()

# height = pd.to_numeric(temp.columns).to_numpy()

# %%
