import pickle

import numpy as np

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

select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]
# select_dates = [datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]

fig, axs = plt.subplots(2, 3, figsize=(9, 6))
ax_ravel = axs.ravel()
idx = 0
plume_schemes = ["obs", "Briggs", "FEPS", "SEV", "Freitas", "WRF-SFIRE"]
markers = ['o', 's', '^', 'D', '*', 'x', 'P', 'v']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
# colors = [
#     '#000000',  # Black
#     '#E69F00',  # Orange
#     '#56B4E9',  # Sky blue
#     '#009E73',  # Bluish green
#     '#F0E442',  # Yellow
#     '#D55E00',  # Vermillion
#     '#CC79A7',  # Reddish purple
#     '#0072B2'  # Blue
# ]

combined_res = {}
for plume_scheme in plume_schemes:
    cur_res = []
    for select_date in select_dates:
        cur_res.append(conc[select_date][plume_scheme])
    cur_res = np.concatenate(cur_res)
    combined_res[plume_scheme] = cur_res

intervals = [100, 150, 200, 50, 50]
for i in range(1, len(plume_schemes)):
    obs_val, model_val = combined_res["obs"], combined_res[plume_schemes[i]]
    ax = ax_ravel[idx]
    for select_date in select_dates:
        t_idx = select_dates.index(select_date)
        ax.scatter(conc[select_date]["obs"], conc[select_date][plume_schemes[i]], marker=markers[t_idx], color=colors[t_idx])
    obs_val, model_val = obs_val.reshape(-1, 1), model_val.reshape(-1, 1)
    plotComparisonNoIntercept(obs_val, model_val, ax)
    ax.set_xlabel("Observations $PM_{2.5}$ ($\mu g/m^3$)")
    ax.set_ylabel("%s $PM_{2.5}$ ($\mu g/m^3$)" % plume_schemes[i])
    max_value = np.max([np.max(obs_val), np.max(model_val)])
    min_value = np.min([np.min(obs_val), np.min(model_val)])
    ax.plot(np.arange(0, max_value + 5, 1), np.arange(0, max_value + 5, 1), '--k')
    ax.set_aspect('equal')
    ax.set_title(plume_schemes[i])
    ax.set_xticks(np.arange(0, max_value + intervals[idx], intervals[idx]))
    ax.set_yticks(np.arange(0, max_value + intervals[idx], intervals[idx]))
    ax.set_xlim(xmax=max_value + 50)
    ax.set_ylim(ymax=max_value + 50)
    idx += 1

ax_ravel[5].set_axis_off()
# Create a scatter plot with markers and colors
for i in range(0, len(select_dates)):
    ax_ravel[5].scatter([], [], marker=markers[i], color=colors[i], label="%s" % select_dates[i].strftime("%y%m%d"))
ax_ravel[5].legend(bbox_to_anchor=(0.2, 1), loc='upper center')
plt.tight_layout()
plt.show()