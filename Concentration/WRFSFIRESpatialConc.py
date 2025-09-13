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
from MonitorInfo.MonitorStyle import monitor_style
import imageio

# select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
#                 datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
case_name = "load_increase"
select_dates = [datetime(2022, 3, 5)]
cmap = discrete_conc_cmap(20)
for select_date in select_dates:
    # sfire_file = "/Volumes/DataStorage/SERDP/data/SFIRE/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y%m%d"), select_date.strftime("%Y-%m-%d"))
    sfire_file = "/Volumes/PubData/FuelSens/%s/wrfout_d01_%s_00:00:00" % (case_name, select_date.strftime("%Y-%m-%d"))
    fort_boundary_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fort_boundary/fort_boundary.json"
    # fire_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/Select_FtBn_BurnInfo.json"
    fire_file = "/Volumes/Shield/WindUncertaintyImpacts/data/BurnInfo/FtStwrt_BurnInfo.json"
    # conc_filename = "/Volumes/DataStorage/SERDP/data/conc/combined_PM25_conc.csv"
    conc_filename = "/Volumes/PubData/WindUncertainty/Measurements/conc/combined_PM25_conc.csv"
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
    for current_time_idx in range(start_time_idx, len(wrf_info["time"])):
        current_time = wrf_info["time"][current_time_idx]
        current_u10 = model_u10[current_time_idx, :, :]
        current_v10 = model_v10[current_time_idx, :, :]
        # plot figure
        fig, ax = plt.subplots(figsize=(5, 4))
        cur_conc = np.squeeze(sfire_ds[specie_name][:][current_time_idx, 0, :, :])
        c = ax.pcolor(wrf_info["Lon"], wrf_info["Lat"], cur_conc, vmin=20, vmax=100, cmap=cmap, shading='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(c, cax=cax, orientation='vertical', extend='both')
        cb.ax.set_ylabel('$PM_{2.5}$ ($\mu g/m^3$)\n', rotation=270, fontsize=12, labelpad=15)
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
                                label=current_monitor, c=monitor_style[current_monitor]["color"],
                                marker=monitor_style[current_monitor]["MarkerStyle"], edgecolors='k', s=60)
            else:
                sc = ax.scatter([monitor_locations[current_monitor][0][0]],
                                [monitor_locations[current_monitor][0][1]],
                                label=current_monitor, c=monitor_style[current_monitor]["color"],
                                marker=monitor_style[current_monitor]["MarkerStyle"], edgecolors='k', s=60)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-81.93, -81.76])
        ax.set_ylim([31.96, 32.09])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(ncol=3, loc='lower center', frameon=False)
        ax.set_title("WRF-SFIRE $PM_{2.5}$ ($\mu g/m^3$)\nUTC: %s" % wrf_info["time"][current_time_idx].strftime("%Y-%m-%d %H:%M"), fontsize=16)
        plt.tight_layout()
        # plt.show()

        fig_name = "/Volumes/PubData/ModelComparisons/work_dir/" + specie_name + "_" + current_time.strftime("%Y-%m-%d_%H:%M") + ".png"
        plt.savefig(fig_name)
        filenames.append(fig_name)
        plt.clf()
        plt.close()

    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    duration_time = 1
    # Save them as frames into a gif
    exportname = "/Volumes/PubData/ModelComparisons/Figures/sfire_conc_%s.gif" % (case_name)
    kargs = {'duration': duration_time}
    imageio.mimsave(exportname, frames, 'GIF', **kargs)

    # clean up work dir
    os.system("rm /Volumes/PubData/ModelComparisons/work_dir/*")
