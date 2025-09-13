import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from utils import CMAQGridInfo, WRFGridInfo, smoke_concentration


def conc_at_obs_CMAQ(cmaq_ds, pollutant_name, obs_lon, obs_lat):
    cmaq_info = CMAQGridInfo(cmaq_ds)
    lat = cmaq_info["Lat"]
    lon = cmaq_info["Lon"]
    pollutant = cmaq_ds[pollutant_name]
    pollutant_surface = pollutant[:][:, 0, :, :]
    # Nearest grid
    distance = (lat - obs_lat) ** 2 + (lon - obs_lon) ** 2
    x, y = np.where(distance == np.min(distance))
    pollutant_surface_at_obs = pollutant_surface[:, x, y]
    pollutant_surface_at_obs = pollutant_surface_at_obs.flatten()
    return {
        "time": cmaq_info["time"],
        "conc": pollutant_surface_at_obs
    }


def conc_at_obs_WRF(wrf_ds, pollutant_name, obs_lon, obs_lat):
    wrf_info = WRFGridInfo(wrf_ds)
    lat = wrf_info["Lat"]
    lon = wrf_info["Lon"]
    # pollutant = wrf_ds[pollutant_name]
    pollutant = smoke_concentration(pollutant_name, wrf_ds)
    pollutant_surface = pollutant[:][:, 0, :, :]
    # Nearest grid
    distance = (lat - obs_lat) ** 2 + (lon - obs_lon) ** 2
    x, y = np.where(distance == np.min(distance))
    pollutant_surface_at_obs = pollutant_surface[:, x, y]
    pollutant_surface_at_obs = pollutant_surface_at_obs.flatten()
    return {
        "time": wrf_info["time"],
        "conc": pollutant_surface_at_obs
    }
