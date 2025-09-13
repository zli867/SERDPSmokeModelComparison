import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from shapely.geometry import shape, MultiPolygon
import pickle
from PlumeHeight.ExtractPlumeHeight import get_bsp_interpolated_plume_height
import numpy as np
from matplotlib.dates import DateFormatter
import pandas as pd
# fuel_name: area
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
plume_schemes = ["SEV", "Freitas"]
scheme_colors = ["red", "blue", "green"]
name_switches = {"N20_VIIRS": "NOAA-20 VIIRS", "N_VIIRS": "NPP VIIRS", "Aqua_MODIS": "Aqua MODIS", "Terra_MODIS": "Terra MODIS"}
# Set figure size
fig, axs = plt.subplots(2, 1, figsize=(12, 6))
ax_ravel = axs.ravel()
fig_idx = 0
for plume_scheme in plume_schemes:
    plume_dict = {}
    for select_date in select_dates:
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
        bsp_file = "/Volumes/DataStorage/SERDP/data/BlueSky/FRP/FtBn%s_%s_FRP_out.json" % (select_date.strftime("%Y_%m_%d"), plume_scheme)
        bsp_plume = get_bsp_interpolated_plume_height(bsp_file)
        for fire_id in bsp_plume.keys():
            fire_name, product_name = fire_id.split("@")[0], fire_id.split("@")[1]
            if fire_name not in plume_dict:
                plume_dict[fire_name] = {product_name: np.mean(bsp_plume[fire_id]["plume_top"])}
            else:
                plume_dict[fire_name][product_name] = np.mean(bsp_plume[fire_id]["plume_top"])
    # plot
    data_list = []
    for location, measurements in plume_dict.items():
        if len(measurements) == 1:
            continue
        for satellite, value in measurements.items():
            data_list.append({
                'Location': location,
                'Satellite': satellite,
                'Value': value
            })

    df = pd.DataFrame(data_list)

    # Calculate bar positions
    locations = df['Location'].unique()
    satellites = df['Satellite'].unique()
    x = np.arange(len(locations))
    width = 0.2  # Width of the bars
    n_satellites = len(satellites)
    ax = ax_ravel[fig_idx]
    # Create grouped bars
    for i, satellite in enumerate(satellites):
        satellite_data = df[df['Satellite'] == satellite]
        offset = width * (i - (n_satellites - 1) / 2)
        ax.bar(x + offset,
                satellite_data.set_index('Location').reindex(locations)['Value'],
                width,
                label=name_switches[satellite])
    locations = [l.split("_")[1] for l in locations]
    # Customize the plot
    ax.set_ylabel('Plume Height (m)', fontsize=12)
    ax.set_title('%s' % plume_scheme, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(locations, rotation=45, ha='right')
    ax.legend(loc="upper right", frameon=False)

    fig_idx += 1
    # stds = []
    # for burn_unit in plume_dict.keys():
    #     cur_height = []
    #     for sat in plume_dict[burn_unit].keys():
    #         cur_height.append(plume_dict[burn_unit][sat])
    #     stds.append((max(cur_height) - min(cur_height)) / min(cur_height))
    # print(plume_scheme)
    # for s in stds:
    #     print(s)
    # print(plume_scheme)
    # print(np.mean(stds))
plt.tight_layout()
plt.show()