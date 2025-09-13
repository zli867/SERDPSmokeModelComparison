import pickle
from utils import plotComparisonNoIntercept
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from MonitorInfo.MonitorStyle import monitor_style
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

plt.rcParams['font.size'] = 12  # Set the default font size to 12 for all plots

with open('/Volumes/Shield/ModelComparisons/stats_data/conc_sec_prim.pickle', 'rb') as f:
    conc = pickle.load(f)

select_dates = [datetime(2021, 3, 20), datetime(2021, 3, 22), datetime(2021, 4, 7), datetime(2021, 4, 20),
                datetime(2022, 3, 27), datetime(2022, 4, 23), datetime(2022, 4, 24), datetime(2022, 4, 25)]

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
plume_schemes = ["Briggs", "FEPS", "SEV", "Freitas"]
# scheme_colors = ["red", "blue", "green", "yellow", "black"]
scheme_colors = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#009E73',  # Bluish green
    '#F0E442',  # Yellow
    '#000000',  # Black
]
for i in range(0, len(plume_schemes)):
    fig, axs = plt.subplots(4, 4, figsize=(12, 8))
    ax_ravel = axs.ravel()
    idx = 0
    for select_date in select_dates:
        for monitor_name in peak_monitors[select_date].keys():
            ax = ax_ravel[idx]
            cur_conc = conc[select_date][monitor_name]
            cur_color = monitor_style[monitor_name]["color"]
            ax.plot(cur_conc[plume_schemes[i]]["time"], cur_conc[plume_schemes[i]]["sec"], color='r', label=plume_schemes[i])
            ax.plot(cur_conc[plume_schemes[i]]["time"], cur_conc[plume_schemes[i]]["prim"], color='g', label=plume_schemes[i], marker='*')
            ax.plot(cur_conc[plume_schemes[i]]["time"], cur_conc[plume_schemes[i]]["conc"], color='k', label=plume_schemes[i], marker='v')
            datefmt = DateFormatter("%H")
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_formatter(datefmt)
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
            ax.set_xlim(xmin=fire_time[select_date], xmax=select_date + timedelta(hours=24))
            ax.set_title("%s %s" % (select_date.strftime("%Y%m%d"), monitor_name))
            idx += 1
    # fig.supxlabel('UTC Hour', fontsize=16)
    # fig.supylabel("%s Concentration ($\mu g/m^3$)" % plume_schemes[i], fontsize=16)
    # Optimize layout with proper spacing
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.15, top=0.92,
                        wspace=0.25, hspace=0.35)

    # Add clear labels
    fig.supxlabel('UTC Hour', fontsize=16, y=0.04)
    fig.supylabel(f'{plume_schemes[i]} Concentration ($\\mu g/m^3$)',
                  fontsize=16, x=0.02)

    # Create clean legend
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, marker='v',
               markersize=6, label='$PM_{2.5}$ Total'),
        Line2D([0], [0], color='red', linewidth=2,
               label='$PM_{2.5}$ Secondary'),
        Line2D([0], [0], color='green', linewidth=2, marker='*',
               markersize=8, label='$PM_{2.5}$ Primary')
    ]

    lgd = fig.legend(handles=legend_elements, loc='lower center',
                     bbox_to_anchor=(0.5, -0.01), ncol=3,
                     frameon=False, fontsize=13)
    plt.tight_layout()
    plt.show()
    # fig.savefig('%s.png' % plume_schemes[i], bbox_extra_artists=(lgd,), bbox_inches='tight')