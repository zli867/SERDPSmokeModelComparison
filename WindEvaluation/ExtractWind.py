import numpy as np


def adjust_ratio(current_height, surface_height):
    # For open terrain (grassland) the typical range is 0.01-0.05 m
    z0 = 0.03
    d = 0
    ratio = np.log((surface_height - d)/z0)/np.log((current_height - d)/z0)
    return ratio


def wind_2_uv(wdspd, wddir):
    u = -wdspd * np.sin(np.deg2rad(wddir))
    v = -wdspd * np.cos(np.deg2rad(wddir))
    return u, v


def uv_2_wind(u, v):
    wdspd = np.sqrt(u**2 + v**2)
    wddir = np.mod(180+np.rad2deg(np.arctan2(u, v)), 360)
    return wdspd, wddir


def windDirectionTransfer(dir1, dir2):
    # Transfer dir1, dir2 to minimum diff
    # Notice that dir1 value will not be changed
    for i in range(0, len(dir1)):
        if np.isnan(dir1[i]) or np.isnan(dir2[i]):
            continue
        else:
            diff = dir1[i] - dir2[i]
            if diff < -180:
                dir2[i] = dir2[i] -360
            elif diff > 180:
                dir2[i] = dir2[i] + 360
    return dir1, dir2


def wrf_wind_uv_2(ds, monitor_lon, monitor_lat):
    lat = ds["XLAT"][:]
    lon = ds["XLONG"][:]
    u_10 = ds["U10"][:]
    v_10 = ds["V10"][:]
    lat = lat[0, :, :]
    lon = lon[0, :, :]
    distance = (lat - monitor_lat) ** 2 + (lon - monitor_lon) ** 2
    x, y = np.where(distance == np.min(distance))
    u_2_at_obs = u_10[:, x, y].flatten() * adjust_ratio(10, 2)
    v_2_at_obs = v_10[:, x, y].flatten() * adjust_ratio(10, 2)
    return u_2_at_obs, v_2_at_obs
