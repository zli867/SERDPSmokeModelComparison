import os
import pandas as pd
from datetime import datetime, timedelta
import json
from shapely.geometry import shape
from utils import plotPolygons, discrete_conc_cmap
import matplotlib.pyplot as plt
import netCDF4 as nc
from utils import WRFGridInfo
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from utils import smoke_concentration

select_date = datetime(2024, 2, 10)
cmap = discrete_conc_cmap(20)
sfire_file = "/Volumes/SERDP/SFIRECMAQ/sfire/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y-%m-%d"))
fort_boundary_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fort_boundary/fort_boundary.json"
fire_file = "/Volumes/PubData/FtSt2024/FtStwrt24_BurnInfo.json"
conc_filename = "/Volumes/PubData/FtSt2024/obs/Trailer_2024_FtSt_PM.csv"
specie_name = "tr17_2"
sfire_ds = nc.Dataset(sfire_file)
model_u10 = sfire_ds["U10"][:]
model_v10 = sfire_ds["V10"][:]

with open(fire_file) as json_file:
    fire_events = json.load(json_file)

with open(fort_boundary_file) as json_file:
    boundary = json.load(json_file)

fort_boundary = shape(boundary["Fort Stewart"])

select_fire_events = []
for fire_event in fire_events["fires"]:
    fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
    if fire_date == select_date:
        select_fire_events.append(fire_event)
rx_polygons = []

fire_start_time = None
for select_fire_event in select_fire_events:
    # rx polygons
    if select_fire_event["type"] == "rx":
        rx_polygons.append(shape(select_fire_event["perimeter"]))
        if fire_start_time is None:
            fire_start_time = datetime.strptime(select_fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S")
        else:
            fire_start_time = min(fire_start_time,
                                  datetime.strptime(select_fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S"))
fire_start_time = datetime(fire_start_time.year, fire_start_time.month, fire_start_time.day, fire_start_time.hour - 1)
monitor_locations = {}
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)
df = df[(df["UTC_time"] >= select_date) & (df["UTC_time"] < select_date + timedelta(days=1))]
for idx, row in df.iterrows():
    if row["monitor"] not in monitor_locations.keys():
        monitor_locations[row["monitor"]] = [(row["lon"], row["lat"])]
    else:
        if (row["lon"], row["lat"]) not in monitor_locations[row["monitor"]]:
            monitor_locations[row["monitor"]].append((row["lon"], row["lat"]))

wrf_info = WRFGridInfo(sfire_ds)
start_time_idx = wrf_info["time"].index(fire_start_time)
filenames = []
current_time_idx = 58
current_time = wrf_info["time"][current_time_idx]
current_u10 = model_u10[current_time_idx, :, :]
current_v10 = model_v10[current_time_idx, :, :]
# plot figure
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# Figure 1
ax_ravel = axs.ravel()
ax = ax_ravel[0]
# cur_conc = np.squeeze(sfire_ds[specie_name][:][current_time_idx, 0, :, :])
cur_conc = smoke_concentration(specie_name, sfire_ds)[:][current_time_idx, 0, :, :]
# cur_conc = np.squeeze(sfire_ds[specie_name][:][current_time_idx, :, :])
c = ax.pcolor(wrf_info["Lon"], wrf_info["Lat"], cur_conc, vmin=0, vmax=200, cmap=cmap, shading='nearest')
# c = ax.pcolor(wrf_info["Lon"], wrf_info["Lat"], cur_conc, cmap=cmap, shading='nearest')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(c, cax=cax, orientation='vertical', extend='both')
cb.ax.set_ylabel('$PM_{2.5}$ ($\mu g/m^3$)\n', rotation=270, fontsize=12, labelpad=15)
# cb.ax.set_ylabel('FUEL FRAC\n', rotation=270, fontsize=12, labelpad=15)
# wind vector
ax.quiver(wrf_info["Lon"][::6, ::6], wrf_info["Lat"][::6, ::6], current_u10[::6, ::6],
          current_v10[::6, ::6], headlength=4, headwidth=5, headaxislength=2)
# fort boundary
plotPolygons([fort_boundary], ax, "black")
# rx
plotPolygons(rx_polygons, ax, "black")
# monitor
for current_monitor in monitor_locations.keys():
    if len(monitor_locations[current_monitor]) > 1:
        sc = ax.scatter([monitor_locations[current_monitor][1][0]],
                        [monitor_locations[current_monitor][1][1]],
                        label=current_monitor, edgecolors='k', s=60)
    else:
        sc = ax.scatter([monitor_locations[current_monitor][0][0]],
                        [monitor_locations[current_monitor][0][1]],
                        label=current_monitor, edgecolors='k', s=60)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-81.85, -81.70])
ax.set_ylim([31.97, 32.07])
ax.set_xticks([])
ax.set_yticks([])
ax.legend(ncol=3, loc='lower center', frameon=False)
ax.set_title("WRF-SFIRE $PM_{2.5}$ ($\mu g/m^3$)\nUTC: %s" % wrf_info["time"][current_time_idx].strftime("%Y-%m-%d %H:%M"), fontsize=16)

# Ignition
ax = ax_ravel[1]
ignition_info = {}
ignition_info[select_date] = {
    "Polygon": [],
    "lon": [],
    "lat": [],
    "time": []
}

select_fire_events = []
for fire_event in fire_events["fires"]:
    fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
    if fire_date == select_date:
        select_fire_events.append(fire_event)

for select_fire_event in select_fire_events:
    # rx polygons
    if select_fire_event["type"] == "rx":
        ignition_info[select_date]["Polygon"].append(shape(select_fire_event["perimeter"]))
        ignition_lon, ignition_lat = select_fire_event["ignition_patterns"][0]["ignition_lng"], select_fire_event["ignition_patterns"][0]["ignition_lat"]
        ignition_time = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in select_fire_event["ignition_patterns"][0]["ignition_time"]]
        ignition_info[select_date]["lat"] = ignition_lat
        ignition_info[select_date]["lon"] = ignition_lon
        ignition_info[select_date]["time"] = ignition_time


cur_info = ignition_info[select_date]
delta_time = np.array(cur_info["time"]) - cur_info["time"][0]
delta_time_sec = [i.seconds for i in delta_time][:-1]
cmap = plt.cm.get_cmap('jet')
norm = Normalize(vmin=min(delta_time_sec), vmax=max(delta_time_sec))

plotPolygons(cur_info["Polygon"], ax, "black")
sc = ax.scatter(cur_info["lon"][:-1], cur_info["lat"][:-1], c=delta_time_sec, s=10, zorder=10, cmap=cmap)
for i in range(0, len(delta_time_sec) - 1):
    cur_line_sec = np.linspace(delta_time_sec[i], delta_time_sec[i + 1], num=2)
    cur_line_lons = np.linspace(cur_info["lon"][i], cur_info["lon"][i + 1], num=2)
    cur_line_lats = np.linspace(cur_info["lat"][i], cur_info["lat"][i + 1], num=2)
    for k in range(0, len(cur_line_sec) - 1):
        rgba = cmap(norm((cur_line_sec[k] + cur_line_sec[k + 1]) / 2))
        ax.plot([cur_line_lons[k], cur_line_lons[k + 1]], [cur_line_lats[k], cur_line_lats[k + 1]], color=rgba)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax=cax1)
cbar.ax.set_ylabel('seconds from ignition start time', rotation=270, labelpad=15, fontsize=12)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Ignition Start Time\nUTC %s" % datetime.strftime(cur_info["time"][0], "%Y-%m-%d %H:%M:%S"), fontsize=16)
# plt.tight_layout()
plt.show()