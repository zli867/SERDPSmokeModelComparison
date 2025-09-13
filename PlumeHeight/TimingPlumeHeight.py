import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from shapely.geometry import shape, MultiPolygon
import pickle
from ExtractPlumeHeight import get_bsp_interpolated_plume_height, get_sfire_plume_height
import numpy as np
from matplotlib.dates import DateFormatter

# fuel_name: area
stats_dict = {}
select_dates = [datetime(2022, 4, 23)]
plume_schemes = ["Briggs", "FEPS", "SEV", "Freitas"]
scheme_colors = ["red", "blue", "green", "yellow"]
for select_date in select_dates:
    print(select_date)
    # get the fire geometry info
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
    lon_min, lat_min, lon_max, lat_max = polys.bounds
    unit_centroid = polys.centroid
    burn_center_lon = unit_centroid.x
    burn_center_lat = unit_centroid.y

    # BlueSky
    bsp_plume_res = {}
    for plume_scheme in plume_schemes:
        bsp_file ="/Volumes/DataStorage/SERDP/data/BlueSky/Timing/FtBn%s_%s_report_out.json" % (select_date.strftime("%Y_%m_%d"), plume_scheme)
        bsp_plume = get_bsp_interpolated_plume_height(bsp_file)
        bsp_plume_res[plume_scheme] = {"plume_top": [], "plume_bottom": []}
        for fire_id in bsp_plume.keys():
            if fire_id in main_fire:
                for key in ["plume_top", "plume_bottom"]:
                    bsp_plume_res[plume_scheme][key].append(np.array(bsp_plume[fire_id][key])[np.newaxis, :])
                bsp_plume_res[plume_scheme]["time"] = bsp_plume[fire_id]["time"]

        for key in ["plume_top", "plume_bottom"]:
            bsp_plume_res[plume_scheme][key] = np.mean(np.concatenate(bsp_plume_res[plume_scheme][key]), axis=0)

    # get obs fuel moisture
    # visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    # BlueSky
    for plume_scheme in plume_schemes:
        idx = plume_schemes.index(plume_scheme)
        ax.fill_between(bsp_plume_res[plume_scheme]["time"], bsp_plume_res[plume_scheme]["plume_top"], bsp_plume_res[plume_scheme]["plume_bottom"], alpha=0.5, color=scheme_colors[idx], label=plume_scheme)
        ax.plot(bsp_plume_res[plume_scheme]["time"], bsp_plume_res[plume_scheme]["plume_top"], color=scheme_colors[idx])
        ax.plot(bsp_plume_res[plume_scheme]["time"], bsp_plume_res[plume_scheme]["plume_bottom"], color=scheme_colors[idx])

    plt.ylabel("Plume Height Above Ground (m)")
    plt.xlabel("Time (UTC)")
    plt.legend()
    plt.title("Plume Height Comparisons Date: %s" %(select_date.strftime("%Y-%m-%d")))
    datefmt = DateFormatter("%H:%M")
    plt.xticks(rotation=45, ha='right')
    ax.xaxis.set_major_formatter(datefmt)
    plt.show()
