from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from MonitorInfo.MonitorStyle import monitor_style
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import pickle
import matplotlib.gridspec as gridspec

# Set label size
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

select_date = datetime(2022, 4, 23)

with open('/Volumes/Shield/ModelComparisons/PublicationFigures/timing_visual.pickle', 'rb') as handle:
    visual_data = pickle.load(handle)

scheme_settings = {
    "Briggs": [0, 120, 150, 1250],
    "FEPS": [0, 120, 150, 1200],
    "SEV": [0, 120, 275, 1200],
    "Freitas": [0, 120, 150, 1200],
    "WRF-SFIRE": [0, 120, 210, 360]
}

# List of all schemes - just WRF-SFIRE
schemes = ["WRF-SFIRE"]

# Create figure and gridspec for just one broken axis plot
fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)

# Common x formatting
datefmt = DateFormatter("%H")
xmin, xmax = datetime(2022, 4, 23, 12), datetime(2022, 4, 24, 0)

# Store axes for easy access
axes = {}

# Create broken axis subplot for WRF-SFIRE
scheme = "WRF-SFIRE"

# Create the broken axis pair
ax_high = fig.add_subplot(gs[0, 0])  # Top part
ax_low = fig.add_subplot(gs[1, 0], sharex=ax_high)  # Bottom part
axes[f'{scheme}_high'] = ax_high
axes[f'{scheme}_low'] = ax_low

# Format x-axis
for ax in (ax_high, ax_low):
    ax.xaxis.set_major_formatter(datefmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    ax.set_xlim(xmin, xmax)

# Plot data for WRF-SFIRE
if scheme in visual_data:
    for start_timing in visual_data[scheme].keys():
        ax_low.plot(visual_data[scheme][start_timing]["time"],
                    visual_data[scheme][start_timing]["conc"],
                    label=start_timing)
        ax_high.plot(visual_data[scheme][start_timing]["time"],
                     visual_data[scheme][start_timing]["conc"])

# Plot observations
ax_low.plot(visual_data["obs"]["time"], visual_data["obs"]["conc"],
            label="obs", marker='o', color=monitor_style["T-1293"]["color"],
            linestyle='None')
ax_high.plot(visual_data["obs"]["time"], visual_data["obs"]["conc"],
             marker='o', color=monitor_style["T-1293"]["color"],
             linestyle='None')

# Set y-limits for broken axis
ax_low.set_ylim(scheme_settings[scheme][0], scheme_settings[scheme][1])
ax_high.set_ylim(scheme_settings[scheme][2], scheme_settings[scheme][3])

# Format y-axis
ax_low.yaxis.set_minor_locator(MultipleLocator(50))
ax_high.tick_params(labelbottom=False)  # hide top plot x labels

# Hide spines between axes
ax_high.spines['bottom'].set_visible(False)
ax_low.spines['top'].set_visible(False)
ax_high.tick_params(bottom=False)  # remove ticks at the break
ax_low.tick_params(top=False)

# Diagonal break marks
d = .015  # diagonal size
kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False, linewidth=1)
ax_high.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax_low.transAxes)
ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# Set title for the subplot
ax_high.set_title(scheme)

# # Add legend
# ax_low.legend(loc='upper right')

plt.tight_layout()
plt.show()