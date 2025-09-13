# import matplotlib.pyplot as plt
# import netCDF4 as nc
# import numpy as np
#
#
# filename1 = "/Users/zongrunli/Desktop/SEC/Combined_3D_CCTM_20210320_FtBn1_SEV_Nudged_noninterp_USFS_updated_sec.nc"
# filename2 = "/Users/zongrunli/Desktop/COMBINE_ACONC_v532_intel_FtBnd03_20210320.nc"
#
# ds1 = nc.Dataset(filename1)
# ds2 = nc.Dataset(filename2)
# print(ds1["PM25_TOT"][:, 0, :, :].shape)
# print(ds2["PM25_TOT"].shape)
# diff = ds1["PM25_TOT"][:, 0, :, :] - np.squeeze(ds2["PM25_TOT"])
# print(np.max(np.abs(diff)))
#
# # PM25_Secondary = ASOMIJ + PM25_SO4 + PM25_NO3 + PM25_NH4 + PM25_CL
# sec2 = ds2["ASOMIJ"][:]
# sec2 = np.squeeze(sec2)
# ratio = sec2 / ds2["PM25_OM"][:, 0, :, :]
#
# # sec1 = ds1["PM25_SEC"][:, 0, :, :]
# # diff2 = sec2 - sec1
# # print(np.max(np.abs(diff2)))
# #
# # t_s = np.mean(ratio, axis=(1, 2))
# # plt.plot(t_s)
# # plt.show()
# for t in range(10, ratio.shape[0]):
#     plt.pcolor(ratio[t, :, :])
#     print(np.min(ratio[t, :, :]))
#     plt.colorbar()
#     plt.show()

import os
import pandas as pd
from datetime import datetime, timedelta
import json
from shapely.geometry import shape
from utils import plotPolygons, discrete_conc_cmap
import matplotlib.pyplot as plt
import netCDF4 as nc
from utils import CMAQGridInfo, WRFGridInfo, smoke_concentration
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from MonitorInfo.MonitorStyle import monitor_style
import imageio

select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]

for select_date in select_dates:
    schemes = ["SEV", "Briggs", "Freitas", "FEPS"]
    cmap = discrete_conc_cmap(20)
    fort_boundary_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fort_boundary/fort_boundary.json"
    fire_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/Select_FtBn_BurnInfo.json"
    conc_filename = "/Volumes/DataStorage/SERDP/data/conc/combined_PM25_conc.csv"

    with open(fire_file) as json_file:
        fire_events = json.load(json_file)

    with open(fort_boundary_file) as json_file:
        boundary = json.load(json_file)

    fort_boundary = shape(boundary["Fort Benning"])

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

    model_file = "/Volumes/DataStorage/SERDP/data/CMAQ/Combined_3D_CCTM_%s_FtBn1_Briggs_Nudged_noninterp_USFS_updated.nc" % (datetime.strftime(select_date, "%Y%m%d"))
    ds = nc.Dataset(model_file)
    cmaq_info = CMAQGridInfo(ds)
    start_time_idx = cmaq_info["time"].index(fire_start_time)
    filenames = []

    for current_time_idx in range(start_time_idx, len(cmaq_info["time"])):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        cmaq_mapping = {0: 0, 1: 1, 2: 3, 3: 4}
        current_time = cmaq_info["time"][current_time_idx]
        # plot figure
        ax_ravel = axs.ravel()
        fig_idx = 0
        for i in range(0, 4):
            cur_scheme = schemes[i]
            cur_model_file = "/Volumes/PubData/ModelComparisons/SOA/COMBINE_ACONC_v532_intel_FtBnd03_%s_%s.nc" % (cur_scheme, datetime.strftime(select_date, "%Y%m%d"))
            cur_bkg = "/Volumes/PubData/ModelComparisons/SOA/COMBINE_ACONC_v532_intel_FtBnd03_BKG_%s.nc" % (datetime.strftime(select_date, "%Y%m%d"))
            bkg_ds = nc.Dataset(cur_bkg)
            cur_ds = nc.Dataset(cur_model_file)
            smoke_total = np.squeeze(cur_ds["PM25_OM"][:][current_time_idx, 0, :, :]) - np.squeeze(bkg_ds["PM25_OM"][:][current_time_idx, 0, :, :])
            # ASOMIJ + PM25_SO4 + PM25_NO3 + PM25_NH4 + PM25_CL
            # total_sec = np.squeeze(cur_ds["ASOMIJ"][:][current_time_idx, 0, :, :]) + np.squeeze(cur_ds["PM25_SO4"][:][current_time_idx, 0, :, :]) + \
            #             np.squeeze(cur_ds["PM25_NO3"][:][current_time_idx, 0, :, :]) + np.squeeze(cur_ds["PM25_NH4"][:][current_time_idx, 0, :, :]) + np.squeeze(cur_ds["PM25_CL"][:][current_time_idx, 0, :, :])
            # bkg_sec = np.squeeze(bkg_ds["ASOMIJ"][:][current_time_idx, 0, :, :]) + np.squeeze(bkg_ds["PM25_SO4"][:][current_time_idx, 0, :, :]) + \
            #           np.squeeze(bkg_ds["PM25_NO3"][:][current_time_idx, 0, :, :]) + np.squeeze(bkg_ds["PM25_NH4"][:][current_time_idx, 0, :, :]) + np.squeeze(bkg_ds["PM25_CL"][:][current_time_idx, 0, :, :])
            total_sec = np.squeeze(cur_ds["ASOMIJ"][:][current_time_idx, 0, :, :])
            bkg_sec = np.squeeze(bkg_ds["ASOMIJ"][:][current_time_idx, 0, :, :])
            total_prim = np.squeeze(cur_ds["APOMIJ"][:][current_time_idx, 0, :, :])
            bkg_prim = np.squeeze(bkg_ds["APOMIJ"][:][current_time_idx, 0, :, :])
            smoke_sec = total_sec - bkg_sec
            smoke_prim = total_prim - bkg_prim
            cur_ratio = (smoke_prim) / (smoke_total)
            cur_ratio[smoke_total <= 20] = 0
            ax = ax_ravel[fig_idx]
            c = ax.pcolor(cmaq_info["Lon"], cmaq_info["Lat"], cur_ratio, vmin=0, vmax=1, cmap=cmap, shading='nearest')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(c, cax=cax, orientation='vertical', extend='both')
            cb.ax.set_ylabel('$PM_SEC_{2.5}$ / $PM_{2.5}$ \n', rotation=270, fontsize=12, labelpad=15)
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
            ax.set_title("%s" % schemes[i], fontsize=16)
            ax.set_xlim([-85.050, -84.594])
            ax.set_ylim([32.193, 32.579])
            ax.set_xticks([])
            ax.set_yticks([])
            fig_idx += 1
        plt.suptitle("$PM_{2.5}$ ($\mu g/m^3$) UTC: %s" % cmaq_info["time"][current_time_idx].strftime("%Y-%m-%d %H:%M"), fontsize=16)
        plt.tight_layout()
        plt.show()

    #     fig_name = "/Volumes/Shield/ModelComparisons/WebSrc/work_dir/" + specie_name + "_" + current_time.strftime("%Y-%m-%d_%H:%M") + ".png"
    #     plt.savefig(fig_name)
    #     filenames.append(fig_name)
    #     plt.clf()
    #     plt.close()
    #
    # frames = []
    # for filename in filenames:
    #     frames.append(imageio.imread(filename))
    #
    # duration_time = 1
    # # Save them as frames into a gif
    # exportname = "/Volumes/Shield/ModelComparisons/WebSrc/conc_%s.gif" % (datetime.strftime(select_date, "%Y%m%d"))
    # kargs = {'duration': duration_time}
    # imageio.mimsave(exportname, frames, 'GIF', **kargs)
    #
    # # clean up work dir
    # os.system("rm /Volumes/Shield/ModelComparisons/WebSrc/work_dir/*")
