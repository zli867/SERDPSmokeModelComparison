from ExtractConsumption import get_bsp_consumptions, get_sfire_consumptions
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime
import json
from shapely.geometry import shape, MultiPolygon
import pickle

# fuel_name: area
bsp_consumption_total = {}
sfire_consumption_total = {}
stats_dict = {}
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
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

    sfire_file = "/Volumes/DataStorage/SERDP/data/SFIRE/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y%m%d"), select_date.strftime("%Y-%m-%d"))
    bsp_file ="/Volumes/DataStorage/SERDP/data/BlueSky/FtBn%s_Briggs_out.json" % (select_date.strftime("%Y_%m_%d"))
    sfire_ds = nc.Dataset(sfire_file)
    bsp_consumption_info = get_bsp_consumptions(bsp_file)
    sfire_consumption_info = get_sfire_consumptions(sfire_ds, lon_min, lon_max, lat_min, lat_max)

    # stats
    stats_dict[select_date] = {"bsp": 0, "sfire": 0}

    # combine to res
    for fire_name in bsp_consumption_info.keys():
        if fire_name not in main_fire:
            continue
        for fuel_name in bsp_consumption_info[fire_name].keys():
            consumption = bsp_consumption_info[fire_name][fuel_name]
            stats_dict[select_date]["bsp"] += consumption
            if fuel_name in bsp_consumption_total.keys():
                bsp_consumption_total[fuel_name] += consumption
            else:
                bsp_consumption_total[fuel_name] = consumption

    for fuel_name in sfire_consumption_info.keys():
        consumption = sfire_consumption_info[fuel_name]
        stats_dict[select_date]["sfire"] += consumption
        if consumption == 0:
            continue
        if fuel_name in sfire_consumption_total.keys():
            sfire_consumption_total[fuel_name] += consumption
        else:
            sfire_consumption_total[fuel_name] = consumption

print(stats_dict)
fig, axs = plt.subplots(2, 1, figsize=(16, 8))
ax_ravel = axs.ravel()
# bluesky
ax = ax_ravel[0]
fuel_types = list(bsp_consumption_total.keys())
fuel_consumptions = list(bsp_consumption_total.values())
ax.bar(fuel_types, fuel_consumptions, width=0.4)
ax.set_ylabel("Fuel Consumptions (tons)")
ax.set_title("BlueSky Consumed Fuel")
print(fuel_types)
print(fuel_consumptions)
# SFIRE
ax = ax_ravel[1]
fuel_types = list(sfire_consumption_total.keys())
fuel_consumptions = list(sfire_consumption_total.values())
ax.bar(fuel_types, fuel_consumptions, width=0.4)
ax.set_ylabel("Fuel Consumptions (tons)")
ax.tick_params(axis='x', labelrotation=30)
ax.set_title("WRF-SFIRE Consumed Fuel")
print(fuel_types)
print(fuel_consumptions)
plt.tight_layout()
plt.show()
#
# # save stats
# with open('/Volumes/Shield/ModelComparisons/stats_data/consumptions.pickle', 'wb') as handle:
#     pickle.dump(stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# res = {"bsp": bsp_consumption_total, "sfire": sfire_consumption_total}
# with open('/Volumes/Shield/ModelComparisons/stats_data/consumptions_category.pickle', 'wb') as handle:
#     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
