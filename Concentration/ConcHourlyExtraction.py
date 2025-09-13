import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
import numpy as np

with open('/Volumes/Shield/ModelComparisons/stats_data/conc.pickle', 'rb') as f:
    conc = pickle.load(f)

select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
# select_dates = [datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
fire_time = {}
fire_file = "/Volumes/DataStorage/SERDP/data/BurnInfo/Select_BurnInfo.json"
# fire time
with open(fire_file) as json_file:
    fire_events = json.load(json_file)
for fire_event in fire_events["fires"]:
    fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
    fire_start_time = datetime.strptime(fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S")
    fire_start_time = datetime(fire_start_time.year, fire_start_time.month, fire_start_time.day, fire_start_time.hour)
    if fire_date not in fire_time.keys():
        fire_time[fire_date] = fire_start_time
    else:
        fire_time[fire_date] = min(fire_time[fire_date], fire_start_time)

peak_monitors = {
    datetime(2021, 3, 20): {
        "USFS1033": [datetime(2021, 3, 20, 17), datetime(2021, 3, 20, 21)]
    },
    datetime(2021, 3, 22): {
        "USFS1033": [datetime(2021, 3, 22, 15), datetime(2021, 3, 22, 21)]
    },
    datetime(2021, 4, 7): {
        "USFS1033": [datetime(2021, 4, 7, 18), datetime(2021, 4, 7, 23)],
        "USFS1032": [datetime(2021, 4, 7, 18), datetime(2021, 4, 7, 23)]
    },
    datetime(2021, 4, 20): {
        "USFS1033": [datetime(2021, 4, 20, 17), datetime(2021, 4, 20, 23)],
        "USFS1032": [datetime(2021, 4, 20, 17), datetime(2021, 4, 20, 23)]
    },
    datetime(2022, 3, 27): {
        "T-1291": [datetime(2022, 3, 27, 15), datetime(2022, 3, 27, 23)],
        "T-1292": [datetime(2022, 3, 27, 12), datetime(2022, 3, 27, 23)]
    },
    datetime(2022, 4, 23): {
        "T-1293": [datetime(2022, 4, 23, 16), datetime(2022, 4, 23, 23)]
    },
    datetime(2022, 4, 24): {
        "T-1293": [datetime(2022, 4, 24, 15), datetime(2022, 4, 24, 23)],
        "USFS 1079": [datetime(2022, 4, 24, 16), datetime(2022, 4, 24, 23)],
    },
    datetime(2022, 4, 25): {
        "Main-Trailer": [datetime(2022, 4, 25, 18), datetime(2022, 4, 25, 23)],
        "T-1292": [datetime(2022, 4, 25, 19), datetime(2022, 4, 25, 23)],
        "USFS 1078": [datetime(2022, 4, 25, 19), datetime(2022, 4, 25, 23)],
        "T-1293": [datetime(2022, 4, 25, 21), datetime(2022, 4, 25, 23)],
        "USFS 1079": [datetime(2022, 4, 25, 19), datetime(2022, 4, 25, 23)],
    }
}
# peak_monitors = {
#     datetime(2022, 3, 27): {
#         "T-1291": [datetime(2022, 3, 27, 15), datetime(2022, 3, 27, 23)],
#         "T-1292": [datetime(2022, 3, 27, 12), datetime(2022, 3, 27, 23)]
#     },
#     datetime(2022, 4, 23): {
#         "T-1293": [datetime(2022, 4, 23, 16), datetime(2022, 4, 23, 23)]
#     },
#     datetime(2022, 4, 24): {
#         "T-1293": [datetime(2022, 4, 24, 15), datetime(2022, 4, 24, 23)],
#     },
#     datetime(2022, 4, 25): {
#         "Main-Trailer": [datetime(2022, 4, 25, 18), datetime(2022, 4, 25, 23)],
#         "T-1292": [datetime(2022, 4, 25, 19), datetime(2022, 4, 25, 23)],
#         "T-1293": [datetime(2022, 4, 25, 21), datetime(2022, 4, 25, 23)],
#     }
# }
plume_schemes = ["Briggs", "FEPS", "SEV", "Freitas", "WRF-SFIRE"]
hourly_res = {}
for select_date in select_dates:
    for monitor_name in peak_monitors[select_date].keys():
        cur_conc = conc[select_date][monitor_name]
        cur_start_time = fire_time[select_date]
        cur_time = cur_conc["obs"]["time"][cur_conc["obs"]["time"] >= cur_start_time].to_numpy()
        cur_obs = cur_conc["obs"]["conc"][cur_conc["obs"]["time"] >= cur_start_time].to_numpy()
        cur_time = [t.astype('datetime64[s]').tolist() for t in cur_time]
        hourly_res[select_date] = {
            "time": cur_time, "obs": cur_obs
        }
        for plume_scheme in plume_schemes:
            hourly_res[select_date][plume_scheme] = []
        for t in cur_time:
            for plume_scheme in plume_schemes:
                model_time = np.array(cur_conc[plume_scheme]["time"])
                scheme_conc = cur_conc[plume_scheme]["conc"][(model_time >= t) & (model_time < t + timedelta(hours=1))]
                hourly_res[select_date][plume_scheme].append(np.mean(scheme_conc))

        for plume_scheme in plume_schemes:
            hourly_res[select_date][plume_scheme] = np.array(hourly_res[select_date][plume_scheme])

# save to dict
with open('/Volumes/Shield/ModelComparisons/stats_data/conc_hourly_2022_Trailer.pickle', 'wb') as handle:
    pickle.dump(hourly_res, handle, protocol=pickle.HIGHEST_PROTOCOL)