import pickle
from utils import plotComparisonNoIntercept
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.lines as mlines

plt.rcParams['font.size'] = 12  # Set the default font size to 12 for all plots

markers = ['o', 's', '^', 'D', '*', 'x', 'P', 'v']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
ax_ravel = axs.ravel()
# Burned Area
with open('/Volumes/Shield/ModelComparisons/stats_data/burned_area.pickle', 'rb') as f:
    burned_area = pickle.load(f)

ax = ax_ravel[0]
bsp_area, sfire_area = [], []
for dates in burned_area.keys():
    bsp_area.append(burned_area[dates]["bsp"])
    sfire_area.append(burned_area[dates]["sfire"])

# ax.scatter(bsp_area, sfire_area)
for dates in burned_area.keys():
    cur_idx = select_dates.index(dates)
    ax.scatter([burned_area[dates]["bsp"]], [burned_area[dates]["sfire"]],
               color=colors[cur_idx], marker=markers[cur_idx], edgecolors='k', s=100)
bsp_area = np.array(bsp_area).reshape(-1, 1)
sfire_area = np.array(sfire_area).reshape(-1, 1)
plotComparisonNoIntercept(bsp_area, sfire_area, ax)
# ax.set_xlabel("BlueSky Burned Area (acres)")
# ax.set_ylabel("WRF-SFIRE Burned Area (acres)")
ax.set_ylabel("WRF-SFIRE", fontsize=16)
max_value = np.max([np.max(bsp_area), np.max(sfire_area)])
min_value = np.min([np.min(bsp_area), np.min(sfire_area)])
ax.plot(np.arange(0, 2510, 10), np.arange(0, 2510, 10), '--k')
ax.set_xlim(0, 2500)
ax.set_ylim(0, 2500)
ax.set_aspect('equal')
ax.set_title("Burned Areas (unit: acres)", fontsize=16)

# Consumptions
with open('/Volumes/Shield/ModelComparisons/stats_data/consumptions.pickle', 'rb') as f:
    consumptions = pickle.load(f)

ax = ax_ravel[1]
bsp_consumption, sfire_consumption = [], []
for dates in consumptions.keys():
    bsp_consumption.append(consumptions[dates]["bsp"])
    sfire_consumption.append(consumptions[dates]["sfire"])

# ax.scatter(bsp_consumption, sfire_consumption)
for dates in consumptions.keys():
    cur_idx = select_dates.index(dates)
    ax.scatter([consumptions[dates]["bsp"]], [consumptions[dates]["sfire"]],
               color=colors[cur_idx], marker=markers[cur_idx], edgecolors='k', s=100)

bsp_consumption = np.array(bsp_consumption).reshape(-1, 1)
sfire_consumption = np.array(sfire_consumption).reshape(-1, 1)
plotComparisonNoIntercept(bsp_consumption, sfire_consumption, ax)
# ax.set_xlabel("BlueSky Fuel Consumptions (tons)")
# ax.set_ylabel("WRF-SFIRE Fuel Consumptions (tons)")
ax.set_xlabel("BlueSky", fontsize=16)
max_value = np.max([np.max(bsp_consumption), np.max(sfire_consumption)])
min_value = np.min([np.min(bsp_consumption), np.min(sfire_consumption)])
ax.plot(np.arange(0, 11000, 100), np.arange(0, 11000, 100), '--k')
ax.set_xlim(0, 10000)
ax.set_ylim(0, 10000)
ax.set_aspect('equal')
ax.set_title("Fuel Consumptions (unit: tons)", fontsize=16)

# Emissions
with open('/Volumes/Shield/ModelComparisons/stats_data/emissions.pickle', 'rb') as f:
    emissions = pickle.load(f)

ax = ax_ravel[2]
bsp_emissions, sfire_emissions = [], []
for dates in emissions.keys():
    bsp_emissions.append(emissions[dates]["bsp"])
    sfire_emissions.append(emissions[dates]["sfire"])

# ax.scatter(bsp_emissions, sfire_emissions)
for dates in emissions.keys():
    cur_idx = select_dates.index(dates)
    ax.scatter([emissions[dates]["bsp"]], [emissions[dates]["sfire"]],
               color=colors[cur_idx], marker=markers[cur_idx], edgecolors='k', s=100)

bsp_emissions = np.array(bsp_emissions).reshape(-1, 1)
sfire_emissions = np.array(sfire_emissions).reshape(-1, 1)
plotComparisonNoIntercept(bsp_emissions, sfire_emissions, ax)
# ax.set_xlabel("BlueSky Fuel $PM_{2.5}$ Emissions (tons)")
# ax.set_ylabel("WRF-SFIRE Fuel $PM_{2.5}$ Emissions (tons)")
max_value = np.max([np.max(bsp_emissions), np.max(sfire_emissions)])
min_value = np.min([np.min(bsp_emissions), np.min(sfire_emissions)])
ax.plot(np.arange(0, 170, 10), np.arange(0, 170, 10), '--k')
ax.set_xlim(0, 160)
ax.set_ylim(0, 160)
ax.set_xticks(np.arange(0, 160, 50))
ax.set_yticks(np.arange(0, 160, 50))
ax.set_aspect('equal')
ax.set_title("$PM_{2.5}$ Emissions (unit: tons)", fontsize=16)

# Generate legend handles
labels = [select_dates[i].strftime("%y%m%d") for i in range(0, len(select_dates))]  # Example labels
legend_handles = [mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8, label=label)
                  for marker, color, label in zip(markers, colors, labels)]

# Add legend to the figure
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4)

plt.tight_layout()
plt.show()