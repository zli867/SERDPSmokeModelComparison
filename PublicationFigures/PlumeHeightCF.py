import pickle
from matplotlib.lines import Line2D
import netCDF4
from matplotlib.markers import MarkerStyle
from utils import plotComparisonNoInterceptSimple
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from shapely.geometry import shape, MultiPolygon
from utils import findSpatialIndex, WRFGridInfo

with open('/Volumes/Shield/ModelComparisons/stats_data/plume_height.pickle', 'rb') as f:
    plume_heights = pickle.load(f)

# get PBLH
for select_date in plume_heights.keys():
    fire_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/Select_BurnInfo.json"
    with open(fire_file) as json_file:
        fire_events = json.load(json_file)

    cluster_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/BurnCluster.json"
    with open(cluster_file) as json_file:
        cluster_data = json.load(json_file)

    main_fire = cluster_data["BurnCluster"][select_date.strftime("%Y-%m-%d")]["Main"]

    select_fire_events = []
    for fire_event in fire_events["fires"]:
        fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
        if fire_date == select_date and fire_event["id"] in main_fire:
            select_fire_events.append(fire_event)
    rx_polygons = []

    for select_fire_event in select_fire_events:
        # rx polygons
        if select_fire_event["type"] == "rx":
            rx_polygons.append(shape(select_fire_event["perimeter"]))

    # get boundary of ploygons rx_polygons
    polys = MultiPolygon(rx_polygons)
    lon_ctr, lat_ctr = polys.centroid.xy
    lon_ctr, lat_ctr = lon_ctr[0], lat_ctr[0]
    wrf_file = "/Volumes/DataStorage/SERDP/data/WRF/wrfout_d03_%s" % (datetime.strftime(select_date, "%Y-%m-%d_%H:%M:%S"))
    ds = netCDF4.Dataset(wrf_file)
    wrf_info = WRFGridInfo(ds)
    x_idx, y_idx = findSpatialIndex(lon_ctr, lat_ctr, wrf_info["Lon"], wrf_info["Lat"])
    PBLH = ds["PBLH"][:, x_idx, y_idx]
    pbl_info = []
    for cur_t in plume_heights[select_date]["time"]:
        cur_idx = wrf_info["time"].index(cur_t)
        pbl_info.append(PBLH[cur_idx])
    plume_heights[select_date]["PBL"] = pbl_info


plt.rcParams['font.size'] = 12  # Set the default font size to 12 for all plots
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
ax_ravel = axs.ravel()
idx = 0

# red higher than pbl, bottom BSP
for plume_scheme in ["Briggs", "FEPS", "SEV", "Freitas"]:
    bsp_vals, sfire_vals, pbl_vals = [], [], []
    for dates in plume_heights.keys():
        bsp_vals.append(plume_heights[dates][plume_scheme])
        sfire_vals.append(plume_heights[dates]["SFIRE"])
        pbl_vals.append(plume_heights[dates]["PBL"])

    bsp_vals = np.concatenate(bsp_vals).reshape(-1, 1)
    sfire_vals = np.concatenate(sfire_vals).flatten().reshape(-1, 1)
    pbl_vals = np.concatenate(pbl_vals).flatten()
    ax = ax_ravel[idx]
    # red lower than pbl
    # red triangle both higher than pbl, blue square both lower than pbl, orange circle one higher than pbl and one lower
    # both higher
    both_high, both_low, inter = [], [], []
    for i in range(0, len(bsp_vals)):
        if pbl_vals[i] < sfire_vals[i] and pbl_vals[i] < bsp_vals[i]:
            both_high.append(i)
        elif pbl_vals[i] >= sfire_vals[i] and pbl_vals[i] >= bsp_vals[i]:
            both_low.append(i)
        else:
            inter.append(i)
    ax.scatter(bsp_vals[both_high], sfire_vals[both_high], c='r', marker='^')
    ax.scatter(bsp_vals[both_low], sfire_vals[both_low], c='b', marker='s')
    ax.scatter(bsp_vals[inter], sfire_vals[inter], c='orange', marker='o')
    print(plume_scheme)
    print(len(bsp_vals[inter]))
    print(100 * (len(bsp_vals[inter]) / len(bsp_vals)))
    # ax.scatter(bsp_vals, sfire_vals)
    if len(bsp_vals[both_high]) > 2:
        plotComparisonNoInterceptSimple(bsp_vals[both_high], sfire_vals[both_high], ax, 'r')
    if len(bsp_vals[both_low]) > 2:
        plotComparisonNoInterceptSimple(bsp_vals[both_low], sfire_vals[both_low], ax, 'b')
    if len(bsp_vals[inter]) > 2:
        plotComparisonNoInterceptSimple(bsp_vals[inter], sfire_vals[inter], ax, 'orange')
    plotComparisonNoInterceptSimple(bsp_vals, sfire_vals, ax, 'k')
    ax.legend(frameon=False)
    ax.set_xlabel("%s (m)" % plume_scheme, fontsize=16)
    ax.set_ylabel("WRF-SFIRE (m)", fontsize=16)
    # max_value = np.max([np.max(bsp_vals), np.max(sfire_vals)])
    # min_value = np.min([np.min(bsp_vals), np.min(sfire_vals)])
    ax.plot(np.arange(0, 3500, 10), np.arange(0, 3500, 10), '--k')
    ax.set_xticks(np.arange(0, 3200, 1000))
    ax.set_yticks(np.arange(0, 3200, 1000))
    ax.set_xlim(0, 3200)
    ax.set_ylim(0, 3200)
    ax.minorticks_on()
    ax.set_aspect('equal')
    print("%s, Avg: %.2f, Max: %.2f, Min: %.2f" %(plume_scheme, np.mean(bsp_vals), np.max(bsp_vals), np.min(bsp_vals)))
    if idx == 0: print("WRF-SFIRE, Avg: %.2f, Max: %.2f, Min: %.2f" %(np.mean(sfire_vals), np.max(sfire_vals), np.min(sfire_vals)))
    idx += 1

red_triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=16, label='Both > PBL')
blue_square = Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=16, label='Both < PBL')
orange_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=16, label='One > PBL, One < PBL')
legend_handles = [red_triangle, blue_square, orange_circle]
lgd = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3)
plt.suptitle("Plume Height Comparisons (AGL, unit: m)", fontsize=16)
plt.tight_layout()
# plt.show()
fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')