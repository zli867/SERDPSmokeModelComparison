import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from Concentration.ExtractConcentration import conc_at_obs_WRF, WRFGridInfo
from WindEvaluation.ExtractWind import wrf_wind_uv_2
import json
import pickle
from MonitorInfo.MonitorStyle import monitor_style
from matplotlib.ticker import MultipleLocator

select_dates = [datetime(2022, 4, 25)]

conc_filename = "/Volumes/DataStorage/SERDP/data/conc/combined_PM25_conc.csv"
fire_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/Select_BurnInfo.json"
uncertainty_file = "/Volumes/Shield/ModelComparisons/UncertainWindAnalysis/equal_time_adv_prior_lsq.pickle"

with open(uncertainty_file, "rb") as f:
    uncertainty_data = pickle.load(f)

method_name = "Equal Time Back/forward Trajectory (prior knowledge)"
fire_time = {}

monitor_met_map = {
    datetime(2022, 4, 25): {
        'USFS 1079': 'USFS 1079',
        'T-1292': 'T-1292',
        'T-1290': 'T-1290',
        'T-1293': 'RAWS_FtBn',
        'USFS 1078': 'USFS 1078',
        'Main-Trailer': 'USFS 1078'
    }
}

wind_obs = "/Volumes/DataStorage/SERDP/data/met/combined_MET.csv"
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)
met_df = pd.read_csv(wind_obs, parse_dates=['UTC_time'], date_parser=dateparse)

# fire time
with open(fire_file) as json_file:
    fire_events = json.load(json_file)
for fire_event in fire_events["fires"]:
    fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
    fire_start_time = datetime.strptime(fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S")
    fire_start_time = datetime(fire_start_time.year, fire_start_time.month, fire_start_time.day, fire_start_time.hour)
    if fire_date not in fire_time.keys():
        fire_time[fire_date] = fire_start_time
    else:
        fire_time[fire_date] = min(fire_time[fire_date], fire_start_time)

datefmt = DateFormatter("%m/%d %H")
for select_date in select_dates:
    cur_fire_time = fire_time[select_date]
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

    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    ax_ravel = axs.ravel()
    ax_idx = 0
    monitor_idx = 0
    monitor_names = ["USFS 1079", "T-1293", "USFS 1078", "T-1292", "Main-Trailer", "T-1290"]
    intervals = [20, 10, 5, 4, 5, 4]
    for monitor_name in monitor_names:
        print(monitor_name)
        row, col = monitor_idx // 3, monitor_idx % 3
        wrf_sfire_filename = "/Volumes/DataStorage/SERDP/data/SFIRE/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y%m%d"), select_date.strftime("%Y-%m-%d"))
        wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)
        wrf_info = WRFGridInfo(wrf_sfire_ds)

        current_lon, current_lat = monitor_locations[monitor_name][0], monitor_locations[monitor_name][1]
        current_df = select_df[select_df["monitor"] == monitor_name]
        wrf_time = wrf_info["time"]
        wrf_start_index = wrf_time.index(cur_fire_time)

        wrf_conc = conc_at_obs_WRF(wrf_sfire_ds, "tr17_2", current_lon, current_lat)
        current_obs_time = current_df["UTC_time"]
        current_obs_conc = current_df["PM25"]

        plot_row, plot_col = ax_idx // 3, ax_idx % 3
        # plot figure
        markers = ["x", "*", None]
        ax = ax_ravel[ax_idx]
        # WRF-SFIRE
        ax.plot(wrf_conc["time"][wrf_start_index:], wrf_conc["conc"][wrf_start_index:], label="SFIRE", color=monitor_style[monitor_name]["color"])
        ax.plot(current_obs_time, current_obs_conc, label="obs", marker='o', linestyle="--", color=monitor_style[monitor_name]["color"])
        ax.plot(wrf_time[wrf_start_index:], 35 * np.ones(len(wrf_time[wrf_start_index:])), linestyle="--", color="r")
        ax.text(wrf_time[wrf_start_index:][0], 35, '35', color='orange', verticalalignment='bottom')
        ax.plot(uncertainty_data[select_date][monitor_name]["time"],
                uncertainty_data[select_date][monitor_name]["mean"], 'k--')
        ax.fill_between(uncertainty_data[select_date][monitor_name]["time"],
                         uncertainty_data[select_date][monitor_name]["lower"],
                         uncertainty_data[select_date][monitor_name]["upper"], alpha=0.3, color='gray')

        ax.set_xlim([min(wrf_time[wrf_start_index:]), max(wrf_time[wrf_start_index:])])
        # plot met
        cur_met_monitor = monitor_met_map[select_date][monitor_name]
        ymin, ymax = ax.get_ylim()

        current_wind_df = select_wind_df[select_wind_df["monitor"] == cur_met_monitor]
        met_obs_time = current_wind_df["UTC_time"].to_numpy()
        met_obs_wddir = current_wind_df["wddir"].to_numpy()
        met_obs_wdspd = current_wind_df["wdspd"].to_numpy()

        wrf_u, wrf_v = wrf_wind_uv_2(wrf_sfire_ds, current_lon, current_lat)
        # met
        if len(current_wind_df) > 0:
            u = -met_obs_wdspd * np.sin(np.radians(met_obs_wddir))
            v = -met_obs_wdspd * np.cos(np.radians(met_obs_wddir))
            y_value = ymax * np.ones(u.shape)
            # OBS
            if monitor_name == "T-1290":
                Q = ax.quiver(met_obs_time, y_value + 3, u, v, scale=10, scale_units='height',
                              width=0.01, headaxislength=3, angles='uv', color=monitor_style[monitor_name]["color"],
                              linewidths=10)
                qk = ax.quiverkey(Q, 1.1, 0.2, 2, r'$2 \frac{m}{s}$', labelpos='N', coordinates='axes')
            else:
                Q = ax.quiver(met_obs_time, y_value, u, v, scale=10, scale_units='height',
                              width=0.01, headaxislength=3, angles='uv', color=monitor_style[monitor_name]["color"],linewidths=10)
            # qk = ax.quiverkey(Q, 0.9, 0.2, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='axes')
        # WRF
        current_wrf_time = wrf_info["time"][::3]
        current_wrf_u = wrf_u[::3]
        current_wrf_v = wrf_v[::3]
        y_value = ymax * np.ones(current_wrf_u.shape)
        if monitor_name == "T-1290":
            ax.quiver(current_wrf_time, y_value + 3, current_wrf_u, current_wrf_v, scale=10, scale_units='height',
                      width=0.01, headaxislength=3, angles='uv', alpha=0.6, color=monitor_style[monitor_name]["color"],
                      linewidths=10)
        else:
            ax.quiver(current_wrf_time, y_value, current_wrf_u, current_wrf_v, scale=10, scale_units='height',
                          width=0.01, headaxislength=3, angles='uv', alpha=0.6, color=monitor_style[monitor_name]["color"], linewidths=10)

        if monitor_name == "T-1290":
            ax.set_ylim([ymin, ymax * 1.4])
        else:
            ax.set_ylim([ymin, ymax * 1.35])
        y_major_ticks = ax.get_yticks()

        ax.yaxis.set_minor_locator(MultipleLocator(intervals[monitor_names.index(monitor_name)]))
        ax.tick_params(axis='y', which='minor')
        datefmt = DateFormatter("%H")
        ax.xaxis.set_major_formatter(datefmt)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        if (len(ax_ravel) == 6 and row == 1 and col == 1) or (len(ax_ravel) == 3 and row == 0 and col == 1):
            ax.set_xlabel("UTC Hour", fontsize=16)
        if col == 0:
            ax.set_ylabel("$PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
        if monitor_idx == len(ax_ravel) - 1:
            ax.legend(fontsize=12, frameon=False)
        ax.set_title(monitor_name, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='x', labelrotation=45)
        ax_idx += 1
        monitor_idx += 1
    plt.suptitle(method_name, fontsize=16)
    plt.tight_layout()
    # reserve right margin and place a black key in figure coords
    fig.subplots_adjust(right=0.86)
    plt.show()