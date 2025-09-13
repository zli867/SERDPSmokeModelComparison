from ExtractEmissions import get_bsp_interpolated_emissions, get_sfire_emissions
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from shapely.geometry import shape, MultiPolygon
import netCDF4 as nc
import pickle

stats_dict = {}
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
# select_dates = [datetime(2021, 3, 20)]
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

    end_time = max(sfire_emissions["time"])
    # BlueSky
    # total emissions for each phase
    bsp_total_emissions = {
        "flaming": 0,
        "smoldering": 0,
        "residual": 0
    }
    for fire_id in bsp_emissions.keys():
        if fire_id in main_fire:
            bsp_end_idx = bsp_emissions[fire_id]["time"].index(end_time)
            bsp_total_emissions["flaming"] += np.sum(bsp_emissions[fire_id]["PM2.5"]["flaming"][:bsp_end_idx + 1])
            bsp_total_emissions["smoldering"] += np.sum(bsp_emissions[fire_id]["PM2.5"]["smoldering"][:bsp_end_idx + 1])
            bsp_total_emissions["residual"] += np.sum(bsp_emissions[fire_id]["PM2.5"]["residual"][:bsp_end_idx + 1])

    # WRF-SFIRE
    select_species = ["p25"]
    wrf_sfire_total_emissions = {}
    species_idx = select_species.index("p25")
    wrf_sfire_total_emissions["flaming"] = np.sum(sfire_emissions["p25"][:])

    # # visualization
    # fig = plt.figure(figsize=(8, 6))
    # width = 0.2
    #
    # # BlueSky
    # plt.bar(0 - width/2, bsp_total_emissions["flaming"], width, color='#d31f2b', edgecolor='black', label="Flaming")
    # plt.bar(0 - width/2, bsp_total_emissions["smoldering"], width, bottom=bsp_total_emissions["flaming"], color='#fd7a20', edgecolor='black',
    #         label="Smoldering")
    # plt.bar(0 - width/2, bsp_total_emissions["residual"], width, bottom=bsp_total_emissions["flaming"] + bsp_total_emissions["smoldering"],
    #         color='#2079b3', edgecolor='black', label="Residual")
    #
    # # WRF-SFIRE
    # plt.bar(0 + width/2, wrf_sfire_total_emissions["flaming"], width, color='#d31f2b', edgecolor='black')
    #
    # plt.xticks([0 - width/2, 0 + width/2], ["BlueSky", "WRF-SFIRE"], fontsize=14)
    # plt.xlim(-0.5, 0.5)
    # plt.title("PM2.5 Emission (tons) Date: %s" %(select_date.strftime("%Y-%m-%d")), fontsize=14)
    # plt.legend(fontsize=14)
    # # plt.savefig("/Users/zongrunli/Desktop/work_dir/Emission_" + select_date.strftime("%Y_%m_%d") + ".png", dpi=600,
    # #             bbox_inches='tight')
    # plt.show()

    stats_dict[select_date] = {
        "sfire": wrf_sfire_total_emissions["flaming"],
        "bsp": bsp_total_emissions["residual"] + bsp_total_emissions["smoldering"] + bsp_total_emissions["flaming"]
    }

# save stats
with open('/Volumes/Shield/ModelComparisons/stats_data/emissions.pickle', 'wb') as handle:
    pickle.dump(stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


