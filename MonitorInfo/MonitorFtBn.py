import pandas as pd
from datetime import datetime, timedelta
import json
from shapely.geometry import shape
import matplotlib.pyplot as plt
from MonitorInfo.MonitorStyle import monitor_style
from matplotlib.markers import MarkerStyle
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs


def FIRMSScatter(filename, select_time):
    datepaser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    df = pd.read_csv(filename, parse_dates=["acq_date"], date_parser=datepaser)
    df_select = df[df["acq_date"] == select_time]
    lat_res = df_select["latitude"].to_numpy()
    lon_res = df_select["longitude"].to_numpy()
    return lon_res, lat_res


def plotPolygons(polygon_list, ax, color):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, color, transform=ccrs.PlateCarree())
        else:
            xs, ys = current_polygon.exterior.xy
            ax.plot(xs, ys, color, transform=ccrs.PlateCarree())

select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
# select_dates = [datetime(2021, 3, 20)]
# get the fire geometry info
fort_boundary_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fort_boundary/fort_boundary.json"
fire_file = "/Volumes/Shield/FireFrameworkCF/FtBn/obs_data/fire/FtBn_BurnInfo.json"

with open(fire_file) as json_file:
    fire_events = json.load(json_file)

with open(fort_boundary_file) as json_file:
    boundary = json.load(json_file)

fort_boundary = shape(boundary["Fort Benning"])

for select_date in select_dates:
    select_fire_events = []
    for fire_event in fire_events["fires"]:
        fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
        if fire_date == select_date:
            select_fire_events.append(fire_event)
            print(fire_event["type"])
    rx_polygons = []
    for select_fire_event in select_fire_events:
        # rx polygons
        if select_fire_event["type"] == "rx":
            rx_polygons.append(shape(select_fire_event["perimeter"]))

    monitor_locations = {}
    conc_filename = "/Volumes/Shield/FireFrameworkCF/FtBn/obs_data/conc/FtBn_conc.csv"
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)
    df = df[(df["UTC_time"] >= select_date) & (df["UTC_time"] < select_date + timedelta(days=1))]
    for idx, row in df.iterrows():
        if row["monitor"] not in monitor_locations.keys():
            monitor_locations[row["monitor"]] = [(row["lon"], row["lat"])]
        else:
            if (row["lon"], row["lat"]) not in monitor_locations[row["monitor"]]:
                monitor_locations[row["monitor"]].append((row["lon"], row["lat"]))

    met_filename = "/Volumes/Shield/FireFrameworkCF/FtBn/obs_data/met/FtBn_Met.csv"
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(met_filename, parse_dates=['UTC_time'], date_parser=dateparse)
    df = df[(df["UTC_time"] >= select_date) & (df["UTC_time"] < select_date + timedelta(days=1))]
    for idx, row in df.iterrows():
        if row["monitor"] not in monitor_locations.keys():
            monitor_locations[row["monitor"]] = [(row["lon"], row["lat"])]
        else:
            if (row["lon"], row["lat"]) not in monitor_locations[row["monitor"]]:
                monitor_locations[row["monitor"]].append((row["lon"], row["lat"]))

    # plot figure
    fig1 = plt.figure(figsize=(8, 8))
    request = cimgt.OSM()
    ax = plt.axes(projection=request.crs)
    extent = [-85.06, -84.55, 32.16, 32.6]
    ax.set_extent(extent)
    ax.add_image(request, 10)  # 5 = zoom level

    # fort boundary
    plotPolygons([fort_boundary], ax, "black")
    # rx
    plotPolygons(rx_polygons, ax, "black")
    # FRP
    firms_lon, firms_lat = FIRMSScatter("/Volumes/Shield/FireFrameworkCF/data/FIRMS/FIRMS_combined.csv", select_date)
    ax.scatter(firms_lon, firms_lat, color='r', transform=ccrs.PlateCarree(), s=20)
    # monitor
    scatter_size = 120
    for current_monitor in monitor_locations.keys():
        if len(monitor_locations[current_monitor]) > 1:
            sc = ax.scatter([monitor_locations[current_monitor][1][0]], [monitor_locations[current_monitor][1][1]],
                        label=current_monitor, c=monitor_style[current_monitor]["color"], marker=MarkerStyle(monitor_style[current_monitor]["MarkerStyle"]), transform=ccrs.PlateCarree(), s=scatter_size)
        else:
            sc = ax.scatter([monitor_locations[current_monitor][0][0]], [monitor_locations[current_monitor][0][1]],
                        label=current_monitor, c=monitor_style[current_monitor]["color"], marker=MarkerStyle(monitor_style[current_monitor]["MarkerStyle"]), transform=ccrs.PlateCarree(), s=scatter_size)
        print(current_monitor)
        print([monitor_locations[current_monitor][0][0]], [monitor_locations[current_monitor][0][1]])
    plt.legend(fontsize=12, loc="lower center", ncol=2)
    # plt.title("Fort Benning Monitor Locations: " + select_date.strftime("%Y-%m-%d"))
    # plt.savefig(select_date.strftime("Monitor_%Y_%m_%d") + ".png", dpi=600)
    plt.show()