import os.path
from pyhdf.SD import SD, SDC
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import shapefile
from os import listdir
from os.path import isfile, join
import json
from shapely.geometry import shape
from utils import fillPolygons, StatePolygon, plotPolygons
from shapely.ops import unary_union
def read_data(filename):
    data_field_name = 'Feature_Classification_Flags'
    hdf = SD(filename, SDC.READ)

    res = {}

    # Read lat
    lat_name='Latitude'
    lat_data = hdf.select(lat_name)
    lat = lat_data[:]
    res["lat"] = lat

    # Read lon.
    lon_name='Longitude'
    lon_data = hdf.select(lon_name)
    lon = lon_data[:]
    res["lon"] = lon

    # Read time.
    time_name = 'Profile_UTC_Time'
    time_data = hdf.select(time_name)
    time = time_data[:]
    res["time"] = time

    # Read Day or Night
    day_night_name = "Day_Night_Flag"
    is_night = hdf.select(day_night_name)
    is_night = is_night[:]
    res["is_night"] = is_night

    # Read Mask Data
    data = hdf.select(data_field_name)
    data = data[:]
    res["data"] = data

    # Generate altitude data
    altitude_low = np.arange(-0.5, 8.2, 0.03)
    altitude_mid = np.arange(8.2, 20.2, 0.06)
    altitude_high = np.arange(20.20, 30.09, 0.18)
    res["altitude_low"] = altitude_low
    res["altitude_mid"] = altitude_mid
    res["altitude_high"] = altitude_high

    return res


def covert_UTC_time(time):
    str_time = str(time)
    fraction = time - int(time)
    year = str_time[0:2]
    month = str_time[2:4]
    day = str_time[4:6]
    hour = str(int(24 * fraction))
    hour = hour.rjust(2, '0')
    remain = 24 * fraction - int(24 * fraction)
    minutes = str(int(60 * remain))
    minutes = minutes.rjust(2, '0')
    remain = 60 * remain - int(60 * remain)
    seconds = str(int(60 * remain))
    seconds = seconds.rjust(2, '0')
    date_str = year + "/" + month + "/" + day + " " + hour + ":" + minutes + ":" + seconds
    datetime_object = datetime.strptime(date_str, '%y/%m/%d %H:%M:%S')
    return datetime_object


def read_smoke_traj(filepath):
    res = read_data(filepath)
    lat = res["lat"]
    lon = res["lon"]
    time = res["time"]
    time_obj = [covert_UTC_time(t[0]) for t in time]
    return lon, lat, time_obj


fig, ax = plt.subplots(figsize=(8, 8))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
marked = {}
for select_date in select_dates:
    marked[select_date] = False

# Get all files from data directory
datapath = "/Users/zongrunli/Desktop/Py_CALIPSO/Data/FortBenning"
datafiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
datafiles.sort()
for i in range(0, len(datafiles)):
    if datafiles[i].find("hdf") != -1:
        print(datafiles[i])
        lon, lat, time = read_smoke_traj(os.path.join(datapath, datafiles[i]))
        cur_t = time[0]
        cur_d = datetime(cur_t.year, cur_t.month, cur_t.day)
        cur_color = colors[select_dates.index(cur_d)]
        if marked[cur_d]:
            ax.plot(lon, lat, color=cur_color)
        else:
            if cur_d in [datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]:
                ax.plot(lon, lat, color=cur_color, label=cur_d.strftime("%Y/%m/%d"))
                print(cur_d.strftime("%Y/%m/%d"))
            else:
                ax.plot(lon, lat, color=cur_color)
            marked[cur_d] = True
#    fort_boundary_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fort_boundary/fort_boundary.json"
fort_boundary_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fort_boundary/fort_boundary.json"
with open(fort_boundary_file) as json_file:
    boundary = json.load(json_file)

fort_boundary = shape(boundary["Fort Benning"])
fillPolygons([fort_boundary], ax, "red")
# shp = shapefile.Reader("/Users/zongrunli/Desktop/Py_CALIPSO/Data/GA/Counties_Georgia.shp")
# polys = [shape(rec.shape.__geo_interface__) for rec in shp.shapeRecords()]
# merged = unary_union(polys)
# x, y = merged.exterior.xy
# ax.plot(x, y, 'k', linewidth=1.5)
# plotPolygons(StatePolygon(["Alabama","Florida", "South Carolina","North Carolina","Tennessee"]), ax, 'k')
# xmin, ymin, xmax, ymax = merged.bounds
# ax.set_xlim(xmin - 0.05, xmax + 0.05)
# ax.set_ylim(ymin - 0.05, ymax + 0.05)
ax.legend(fontsize=16, loc='lower left', frameon=False)
ax.tick_params(axis='both', which='major', labelsize=14)  # bigger ticks
ax.set_xlabel("Longitude", fontsize=16)  # x-axis label
ax.set_ylabel("Latitude", fontsize=16)   # y-axis label
ax.set_aspect('equal')
ax.set_title("CALIPSO Scanned Track", fontsize=16)
plt.show()