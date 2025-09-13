import pickle

import numpy as np
import matplotlib.lines as mlines
from utils import plotComparisonNoIntercept
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from MonitorInfo.MonitorStyle import monitor_style
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

plt.rcParams['font.size'] = 12  # Set the default font size to 12 for all plots

with open('/Volumes/Shield/ModelComparisons/stats_data/conc_hourly_2022_Trailer.pickle', 'rb') as f:
    conc = pickle.load(f)


fig, axs = plt.subplots(2, 2, figsize=(7, 8))
ax_ravel = axs.ravel()
idx = 0
plume_schemes = ["Briggs", "FEPS", "SEV", "Freitas", "WRF-SFIRE"]
markers = ['o', 's', '^', 'D', '*', 'x', 'P', 'v']
colors = [
    '#000000',  # Black
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#009E73',  # Bluish green
    '#F0E442',  # Yellow
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish purple
    '#0072B2'  # Blue
]
combined_res = {}
for plume_scheme in plume_schemes:
    cur_res = []
    for select_date in conc.keys():
        cur_res.append(conc[select_date][plume_scheme])
    cur_res = np.concatenate(cur_res)
    combined_res[plume_scheme] = cur_res

select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
labels = [select_dates[i].strftime("%y%m%d") for i in range(0, len(select_dates))]  # Example labels
for i in range(1, len(plume_schemes)):
    obs_val, model_val = combined_res["Briggs"], combined_res[plume_schemes[i]]
    ax = ax_ravel[idx]
    for select_date in select_dates:
        t_idx = select_dates.index(select_date)
        ax.scatter(conc[select_date]["Briggs"], conc[select_date][plume_schemes[i]], marker=markers[t_idx], color=colors[t_idx])
    obs_val, model_val = obs_val.reshape(-1, 1), model_val.reshape(-1, 1)
    plotComparisonNoIntercept(obs_val, model_val, ax)
    ax.set_xlabel("Briggs $PM_{2.5}$ ($\mu g/m^3$)")
    ax.set_ylabel("%s $PM_{2.5}$ ($\mu g/m^3$)" % plume_schemes[i])
    max_value = np.max([np.max(obs_val), np.max(model_val)])
    min_value = np.min([np.min(obs_val), np.min(model_val)])
    ax.plot(np.arange(0, max_value + 5, 1), np.arange(0, max_value + 5, 1), '--k')
    ax.set_xticks(np.arange(0, max_value + 100, 100))
    ax.set_yticks(np.arange(0, max_value + 100, 100))
    ax.set_xlim(0, max_value + 50)
    ax.set_ylim(0, max_value + 50)
    ax.set_aspect('equal')
    # ax.set_title(plume_schemes[i])
    idx += 1

# Create a scatter plot with markers and colors
for i in range(0, len(select_dates)):
    ax_ravel[2].scatter([], [], marker=markers[i], color=colors[i], label="%s" % select_dates[i].strftime("%y%m%d"))
# Generate legend handles
legend_handles = [mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8, label=label)
                  for marker, color, label in zip(markers, colors, labels)]

# Add legend to the figure
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4)
plt.tight_layout()
plt.show()