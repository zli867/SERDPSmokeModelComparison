import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
from ExtractConcentration import conc_at_obs_CMAQ, conc_at_obs_WRF
from WindEvaluation.ExtractWind import wrf_wind_uv_2
from utils import WRFGridInfo
import pickle

res = {}
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
plume_schemes = ["Briggs", "FEPS", "SEV", "Freitas"]
for select_date in select_dates:
    res[select_date] = {}
    fort = "FtBn"
    conc_filename = "/Volumes/DataStorage/SERDP/data/conc/combined_PM25_conc.csv"
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)

    select_df = df[(df["UTC_time"] >= select_date) & (df["UTC_time"] < select_date + timedelta(days=1))]
    select_df = select_df.reset_index(drop=True)
    monitor_names = list(set(select_df["monitor"].to_numpy()))

    monitor_locations = {}
    for monitor_name in monitor_names:
        current_df = select_df[(select_df["monitor"] == monitor_name) & (select_df["UTC_time"] >= select_date + timedelta(hours=15))]
        if len(current_df) > 0:
            monitor_locations[monitor_name] = (current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0])

    monitor_idx = 1
    for monitor_name in monitor_locations.keys():
        res[select_date][monitor_name] = {}
        current_lon = monitor_locations[monitor_name][0]
        current_lat = monitor_locations[monitor_name][1]
        current_df = select_df[select_df["monitor"] == monitor_name]
        current_obs_time = current_df["UTC_time"]
        current_obs_conc = current_df["PM25"]

        # CMAQ
        for plume_scheme in plume_schemes:
            cmaq_filename = "/Volumes/PubData/ModelComparisons/SOA/COMBINE_ACONC_v532_intel_FtBnd03_%s_%s.nc" % (plume_scheme, datetime.strftime(select_date, "%Y%m%d"))
            cur_bkg = "/Volumes/PubData/ModelComparisons/SOA/COMBINE_ACONC_v532_intel_FtBnd03_BKG_%s.nc" % (datetime.strftime(select_date, "%Y%m%d"))
            cmaq_ds = nc.Dataset(cmaq_filename)
            bkg_ds = nc.Dataset(cur_bkg)
            cmaq_conc = conc_at_obs_CMAQ(cmaq_ds, "PM25_TOT", current_lon, current_lat)
            cmaq_conc_sec = conc_at_obs_CMAQ(cmaq_ds, "ASOMIJ", current_lon, current_lat)
            cmaq_conc_prim = conc_at_obs_CMAQ(cmaq_ds, "APOMIJ", current_lon, current_lat)
            cmaq_bkg_conc = conc_at_obs_CMAQ(bkg_ds, "PM25_TOT", current_lon, current_lat)
            cmaq_bkg_conc_sec = conc_at_obs_CMAQ(bkg_ds, "ASOMIJ", current_lon, current_lat)
            cmaq_bkg_conc_prim = conc_at_obs_CMAQ(bkg_ds, "APOMIJ", current_lon, current_lat)
            res[select_date][monitor_name][plume_scheme] = {"time": cmaq_conc["time"], "conc": cmaq_conc["conc"] - cmaq_bkg_conc["conc"],
                                                            "sec": cmaq_conc_sec["conc"] - cmaq_bkg_conc_sec["conc"],
                                                            "prim": cmaq_conc_prim["conc"] - cmaq_bkg_conc_prim["conc"]}

# save stats
with open('/Volumes/Shield/ModelComparisons/stats_data/conc_sec_prim.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)