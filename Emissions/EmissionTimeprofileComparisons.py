import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime
import json
from shapely.geometry import shape, MultiPolygon
import pickle
from ExtractEmissions import get_sfire_emissions, get_bsp_interpolated_emissions
import numpy as np
from matplotlib.dates import DateFormatter

# fuel_name: area
bsp_consumption_total = {}
sfire_consumption_total = {}
stats_dict = {}
select_dates = [datetime(2021, 3, 20)]
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

    sfire_file = "/Volumes/DataStorage/SERDP/data/SFIRE/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y%m%d"), select_date.strftime("%Y-%m-%d"))
    bsp_file ="/Volumes/DataStorage/SERDP/data/BlueSky/FtBn%s_Briggs_out.json" % (select_date.strftime("%Y_%m_%d"))
    sfire_ds = nc.Dataset(sfire_file)

    sfire_species, bsp_species = ["p25"], ["PM2.5"]
    bsp_emissions = get_bsp_interpolated_emissions(bsp_file, bsp_species)
    sfire_emissions = get_sfire_emissions(sfire_ds, sfire_species, lon_min, lon_max, lat_min, lat_max)

    bsp_emission_time = []
    for fire_id in bsp_emissions.keys():
        if fire_id in main_fire:
            bsp_emission_time.extend(bsp_emissions[fire_id]["time"])

    # emission to emission rate (tons/hr) * 3 since 20 min resolution
    # TODO: what should I show?
    bsp_emission_time = list(set(bsp_emission_time))
    bsp_emission_time.sort()
    bsp_flaming = np.zeros(len(bsp_emission_time))
    bsp_smoldering = np.zeros(len(bsp_emission_time))
    bsp_residual = np.zeros(len(bsp_emission_time))
    for fire_id in bsp_emissions.keys():
        if fire_id in main_fire:
            current_time = bsp_emissions[fire_id]["time"]
            for i in range(0, len(current_time)):
                time_idx = bsp_emission_time.index(current_time[i])
                bsp_flaming[time_idx] += bsp_emissions[fire_id]["PM2.5"]["flaming"][i] * 3
                bsp_smoldering[time_idx] += bsp_emissions[fire_id]["PM2.5"]["smoldering"][i] * 3
                bsp_residual[time_idx] += bsp_emissions[fire_id]["PM2.5"]["residual"][i] * 3

    # visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    # BlueSky
    bsp_emission_time.append(bsp_emission_time[-1] + (bsp_emission_time[1] - bsp_emission_time[0]))
    bsp_flaming = np.hstack((bsp_flaming, bsp_flaming[-1]))
    bsp_smoldering = np.hstack((bsp_smoldering, bsp_smoldering[-1]))
    bsp_residual = np.hstack((bsp_residual, bsp_residual[-1]))
    plt.fill_between(bsp_emission_time, bsp_flaming, step='post', label="BlueSky-Flaming", color="#d31f2b", alpha=0.5)
    plt.fill_between(bsp_emission_time, bsp_flaming, bsp_flaming + bsp_smoldering, step='post', label="BlueSky-Smoldering", color="#fd7a20", alpha=0.5)
    plt.fill_between(bsp_emission_time, bsp_flaming + bsp_smoldering, bsp_flaming + bsp_smoldering + bsp_residual, step='post', label="BlueSky-Residual", color="#2079b3", alpha=0.5)

    # WRF-SFIRE
    wrf_sfire_timeprofile = sfire_emissions["p25"] * 3
    wrf_sfire_time = sfire_emissions["time"]
    wrf_sfire_time.append(wrf_sfire_time[-1] + (wrf_sfire_time[1] - wrf_sfire_time[0]))
    wrf_sfire_timeprofile = np.hstack((wrf_sfire_timeprofile, wrf_sfire_timeprofile[-1]))
    plt.step(wrf_sfire_time, wrf_sfire_timeprofile, color="r", lw=2, where='post', label="WRF-SFIRE")

    plt.xlim(min(bsp_emission_time), max(wrf_sfire_time))
    plt.ylabel("Emission Rate (tons/hr)", fontsize=16)
    plt.xlabel("Time (UTC)", fontsize=16)
    plt.legend(fontsize=12)
    # plt.title("PM2.5 Emission Rate Comparisons (tons/hr) Date: %s" %(select_date.strftime("%Y-%m-%d")))
    plt.title("$PM_{2.5}$ Emission Rate Comparisons (tons/hr)", fontsize=16)
    datefmt = DateFormatter("%H:%M")
    plt.xticks(rotation=45, ha='right')
    ax.xaxis.set_major_formatter(datefmt)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()
    # plt.savefig("/Users/zongrunli/Desktop/work_dir/emiss_profile_" + select_date.strftime("%Y_%m_%d") + ".png", dpi=600,
    #             bbox_inches='tight')
