# Experiments Directory
- Author: Maurice
- Date Created: 2026-03-02 12:45
- Last Updated: 2026-03-02 12:45
- Description: Discussion area for Experiments done on the STAQS-campaign dataset
--- 

### Experiments List

### 1. Data Cleaning
- Use an outlier screen process. Look for large non-physical jumps in data from the vertical profile.
- Generate a mask of these values located by height, and time axis

### 2. Removal of Diurnal Cycle
- Time align each curtain / file by time-from-sunrise
- Take the mean, median, and std
- Subtract mean from the curtains 

### 3. Full Vertical Profile
- Take the surface, sonde, TOLNET, Pandora, TEMPO, and GEOS-CF profile and stich it together