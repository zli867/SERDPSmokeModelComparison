import pandas as pd
from datetime import datetime
import netCDF4 as nc
from UncertainWindUtils import get_polygons_ctr, equal_time_trajectory_adv_lsq
import json
from shapely.geometry import shape
import pickle

sfire_uncertainty_res = {}

monitor_met_map = {
    datetime(2022, 4, 25, 0, 0): {
        'USFS 1079': {'coord': [-84.84063, 32.3708], 'met': 'USFS 1079'},
        'T-1292': {'coord': [-84.758889, 32.473222], 'met': 'T-1292'},
        'T-1290': {'coord': [-84.712694, 32.350611], 'met': 'T-1290'},
        'T-1293': {'coord': [-84.877667, 32.364056], 'met': 'RAWS_FtBn'},
        'USFS 1078': {'coord': [-84.77651999999998, 32.45253], 'met': 'USFS 1078'},
        'Main-Trailer': {'coord': [-84.773944, 32.418917], 'met': 'USFS 1078'}
    }
}

for select_date in monitor_met_map.keys():
    print(select_date)
    wrf_sfire_filename = "/Volumes/DataStorage/SERDP/data/SFIRE/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%Y%m%d"), select_date.strftime("%Y-%m-%d"))
    wind_obs = "/Volumes/DataStorage/SERDP/data/met/combined_MET.csv"
    cluster_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/BurnCluster.json"
    fire_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/Select_BurnInfo.json"

    with open(cluster_file) as json_file:
        cluster_data = json.load(json_file)
    main_cluster = cluster_data["BurnCluster"][datetime.strftime(select_date, "%Y-%m-%d")]["Main"]

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    met_df = pd.read_csv(wind_obs, parse_dates=['UTC_time'], date_parser=dateparse)
    wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)

    # fire polygon and fire info
    with open(fire_file) as json_file:
        fire_events = json.load(json_file)

    rx_polygons, fire_start_time = [], []
    for fire_event in fire_events["fires"]:
        fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
        if fire_date == select_date and fire_event["id"] in main_cluster and fire_event["type"] == "rx":
            fire_start_time.append(datetime.strptime(fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S"))
            rx_polygons.append(shape(fire_event["perimeter"]))
    fire_start_time = min(fire_start_time)
    fire_start_hour = datetime(fire_start_time.year, fire_start_time.month, fire_start_time.day, fire_start_time.hour)
    unit_coord_lon, unit_coord_lat = get_polygons_ctr(rx_polygons)
    monitor_mappings = monitor_met_map[select_date]
    fire_obj = {"coord": [unit_coord_lon, unit_coord_lat], "start_time": fire_start_hour}

    uncertainty_res = equal_time_trajectory_adv_lsq(wrf_sfire_ds, met_df, fire_obj, monitor_mappings)
    sfire_uncertainty_res[select_date] = uncertainty_res

with open("/Volumes/Shield/ModelComparisons/UncertainWindAnalysis/equal_time_adv_prior_lsq.pickle", 'wb') as handle:
    pickle.dump(sfire_uncertainty_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
