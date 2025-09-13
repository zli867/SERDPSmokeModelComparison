import pandas as pd
from datetime import datetime, timedelta
import json
from shapely.geometry import shape
import matplotlib.pyplot as plt
from MonitorInfo.MonitorStyle import monitor_style
from matplotlib.markers import MarkerStyle
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs


def plotPolygons(polygon_list, ax, color):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, color)
        else:
            xs, ys = current_polygon.exterior.xy
            ax.plot(xs, ys, color)


def fillPolygons(polygon_list, ax, color):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, color, alpha=0.5)
                ax.plot(xs, ys, 'k')
        else:
            xs, ys = current_polygon.exterior.xy
            ax.fill(xs, ys, color, alpha=0.5)
            ax.plot(xs, ys, 'k')

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
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
burn_units = {}
for select_date in select_dates:
    select_fire_events = []
    rx_polygons = []
    rx_names = []
    for fire_event in fire_events["fires"]:
        fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
        if fire_date == select_date:
            select_fire_events.append(fire_event)
            print(fire_event["type"])
    for select_fire_event in select_fire_events:
        # rx polygons
        if select_fire_event["type"] == "rx":
            rx_polygons.append(shape(select_fire_event["perimeter"]))
            rx_names.append(select_fire_event["id"].split("_")[1])
    burn_units[select_date] = {
        "names": rx_names,
        "polys": rx_polygons
    }

# plot figure
idx = 0
for select_date in select_dates:
    fig, ax = plt.subplots()
    cur_poly = burn_units[select_date]["polys"]
    plotPolygons(cur_poly, ax, colors[idx])
    cur_name = burn_units[select_date]["names"]
    # name
    for i in range(0, len(cur_name)):
        poly = cur_poly[i]
        lon, lat = poly.centroid.xy
        ax.text(lon[0], lat[0], cur_name[i], fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    idx += 1

    plt.show()

fig, ax = plt.subplots()
idx = 0
for select_date in select_dates:
    cur_poly = burn_units[select_date]["polys"]
    plotPolygons(cur_poly, ax, 'k')
    fillPolygons(cur_poly, ax, colors[idx])
    idx += 1

plotPolygons([fort_boundary], ax, 'k')
ax.set_aspect('equal', adjustable='box')
plt.show()