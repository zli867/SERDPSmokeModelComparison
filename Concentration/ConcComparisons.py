import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
from ExtractConcentration import conc_at_obs_CMAQ, conc_at_obs_WRF
import matplotlib.pyplot as plt
from WindEvaluation.ExtractWind import wrf_wind_uv_2, uv_2_wind
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from MonitorInfo.MonitorStyle import monitor_style
from utils import WRFGridInfo, CMAQGridInfo
import matplotlib as mpl
import pickle

res = {}
# {datetime: {"monitor": {}}}
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
plume_schemes = ["Briggs", "FEPS", "SEV", "Freitas"]
scheme_colors = ["red", "blue", "green", "yellow"]
for select_date in select_dates:
    res[select_date] = {}
    fort = "FtBn"
    conc_filename = "/Volumes/DataStorage/SERDP/data/conc/combined_PM25_conc.csv"
    wrf_sfire_filename = "/Volumes/DataStorage/SERDP/data/SFIRE/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y%m%d"), select_date.strftime("%Y-%m-%d"))
    wrf_filename = "/Volumes/DataStorage/SERDP/data/WRF/wrfout_d03_%s" %(datetime.strftime(select_date, "%Y-%m-%d_%H:%M:%S"))
    wind_obs = "/Volumes/DataStorage/SERDP/data/met/combined_MET.csv"

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)
    met_df = pd.read_csv(wind_obs, parse_dates=['UTC_time'], date_parser=dateparse)

    select_df = df[(df["UTC_time"] >= select_date) & (df["UTC_time"] < select_date + timedelta(days=1))]
    select_df = select_df.reset_index(drop=True)
    select_wind_df = met_df[(met_df["UTC_time"] >= select_date) & (met_df["UTC_time"] < select_date + timedelta(days=1))]
    select_wind_df = select_wind_df.reset_index(drop=True)
    monitor_names = list(set(select_df["monitor"].to_numpy()))

    monitor_locations = {}
    for monitor_name in monitor_names:
        current_df = select_df[(select_df["monitor"] == monitor_name) & (select_df["UTC_time"] >= select_date + timedelta(hours=15))]
        if len(current_df) > 0:
            monitor_locations[monitor_name] = (current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0])

    wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)
    wrf_ds = nc.Dataset(wrf_filename)
    wrf_info = WRFGridInfo(wrf_ds)
    monitor_idx = 1
    for monitor_name in monitor_locations.keys():
        res[select_date][monitor_name] = {}
        fig, ax = plt.subplots(figsize=(8, 6))
        current_lon = monitor_locations[monitor_name][0]
        current_lat = monitor_locations[monitor_name][1]
        wrf_conc = conc_at_obs_WRF(wrf_sfire_ds, "tr17_2", current_lon, current_lat)
        current_df = select_df[select_df["monitor"] == monitor_name]
        current_obs_time = current_df["UTC_time"]
        current_obs_conc = current_df["PM25"]

        # meteorological condition
        current_wind_df = select_wind_df[select_wind_df["monitor"] == monitor_name]
        met_obs_time = current_wind_df["UTC_time"].to_numpy()
        met_obs_wddir = current_wind_df["wddir"].to_numpy()
        met_obs_wdspd = current_wind_df["wdspd"].to_numpy()
        wrf_u, wrf_v = wrf_wind_uv_2(wrf_ds, current_lon, current_lat)

        # plot figure
        # CMAQ
        for plume_scheme in plume_schemes:
            cmaq_filename = "/Volumes/DataStorage/SERDP/data/CMAQ/Combined_3D_CCTM_%s_FtBn1_%s_Nudged_noninterp_USFS_updated.nc" % (datetime.strftime(select_date, "%Y%m%d"), plume_scheme)
            cmaq_ds = nc.Dataset(cmaq_filename)
            cmaq_conc = conc_at_obs_CMAQ(cmaq_ds, "PM25_TOT", current_lon, current_lat)
            ax.plot(cmaq_conc["time"], cmaq_conc["conc"], label=plume_scheme, color=scheme_colors[plume_schemes.index(plume_scheme)])
            res[select_date][monitor_name][plume_scheme] = {"time": cmaq_conc["time"], "conc": cmaq_conc["conc"]}

        # WRF-SFIRE
        ax.plot(wrf_conc["time"], wrf_conc["conc"], label="WRF_SFIRE", color='k')
        res[select_date][monitor_name]["WRF-SFIRE"] = {"time": wrf_conc["time"], "conc": wrf_conc["conc"]}
        ax.plot(current_obs_time, current_obs_conc, label=monitor_name, marker='o', color=monitor_style[monitor_name]["color"])
        res[select_date][monitor_name]["obs"] = {"time": current_obs_time, "conc": current_obs_conc}
        ymin, ymax = ax.get_ylim()

        # met
        if len(current_wind_df) > 0:
            u = -met_obs_wdspd * np.sin(np.radians(met_obs_wddir))
            v = -met_obs_wdspd * np.cos(np.radians(met_obs_wddir))
            y_value = ymax * np.ones(u.shape)
            # OBS
            Q = ax.quiver(met_obs_time, y_value, u, v, scale=50, scale_units='height',
                          width=0.004, headaxislength=3, angles='uv', color=monitor_style[monitor_name]["color"])
            # qk = ax.quiverkey(Q, 0.9, 0.2, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
        # WRF
        current_wrf_time = wrf_info["time"][::3]
        current_wrf_u = wrf_u[::3]
        current_wrf_v = wrf_v[::3]
        y_value = ymax * np.ones(current_wrf_u.shape)
        ax.quiver(current_wrf_time, y_value, current_wrf_u, current_wrf_v, scale=50, scale_units='height',
                      width=0.004, headaxislength=3, angles='uv', alpha=0.3, color=monitor_style[monitor_name]["color"])

        plt.legend(fontsize=12, frameon=False)
        datefmt = DateFormatter("%H")
        # plt.xticks(rotation=45, ha='right')
        ax.xaxis.set_major_formatter(datefmt)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        # plt.title(monitor_name, fontsize=16)
        ax.set_xlim(datetime(2021, 3, 20, 15), datetime(2021, 3, 21, 0))
        ax.set_xlabel("UTC Hour", fontsize=16)
        # ax.set_ylabel("$PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
        plt.title("USFS 1033 $PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("conc_%s_%d.png" % (select_date.strftime("%Y_%m_%d"), monitor_idx))
        monitor_idx += 1
        # plt.show()

# save stats
with open('/Volumes/Shield/ModelComparisons/stats_data/conc.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)